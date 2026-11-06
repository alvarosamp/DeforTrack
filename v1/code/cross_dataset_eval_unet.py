import argparse
import csv
import io
import os
import time
import zipfile
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import keras
import numpy as np
from PIL import Image


def read_image(dataset_root, dataset_zip, image_name):
    image_path = dataset_root / "images" / image_name
    if image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        zip_name = f"Forest Segmented/Forest Segmented/images/{image_name}"
        image = Image.open(io.BytesIO(dataset_zip.read(zip_name))).convert("RGB")
    image = image.resize((256, 256))
    return np.asarray(image, dtype=np.float32) / 255.0


def read_mask(dataset_zip, mask_name):
    zip_name = f"Forest Segmented/Forest Segmented/masks/{mask_name}"
    mask = Image.open(io.BytesIO(dataset_zip.read(zip_name))).convert("L")
    mask = mask.resize((256, 256))
    return np.asarray(mask) > 127


def metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    union = tp + fp + fn
    denom_dice = (2 * tp) + fp + fn
    return {
        "iou": float(tp / union) if union else 1.0,
        "dice": float((2 * tp) / denom_dice) if denom_dice else 1.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "pixel_accuracy": float((tp + tn) / (tp + fp + fn + tn)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_root / "meta_data.csv"
    with open(metadata_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    per_image_path = out_dir / "cross_dataset_unet_per_image.csv"
    done = set()
    if args.resume and per_image_path.exists():
        with open(per_image_path, newline="", encoding="utf-8") as fh:
            done = {row["image"] for row in csv.DictReader(fh)}

    model = keras.saving.load_model(args.model, compile=False)
    values = []
    start = time.perf_counter()

    write_header = not per_image_path.exists() or not args.resume
    with open(per_image_path, "a" if args.resume else "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(
            out_fh,
            fieldnames=[
                "model",
                "image",
                "iou",
                "dice",
                "precision",
                "recall",
                "pixel_accuracy",
                "inference_time_s",
            ],
        )
        if write_header:
            writer.writeheader()

        with zipfile.ZipFile(args.dataset_zip) as dataset_zip:
            batch_images, batch_rows, batch_masks = [], [], []
            processed = 0
            for row in rows:
                if row["image"] in done:
                    continue
                batch_images.append(read_image(dataset_root, dataset_zip, row["image"]))
                batch_masks.append(read_mask(dataset_zip, row["mask"]))
                batch_rows.append(row)

                if len(batch_images) == args.batch_size:
                    t0 = time.perf_counter()
                    preds = model.predict(np.stack(batch_images, axis=0), verbose=0)
                    infer_time = (time.perf_counter() - t0) / len(batch_images)
                    for pred_raw, item_row, gt in zip(preds, batch_rows, batch_masks):
                        pred = np.squeeze(pred_raw) >= args.threshold
                        item = metrics(pred, gt)
                        item["model"] = "U-Net"
                        item["image"] = item_row["image"]
                        item["inference_time_s"] = infer_time
                        writer.writerow(item)
                        values.append(item)
                    out_fh.flush()
                    processed += len(batch_images)
                    if args.progress_every and processed % args.progress_every == 0:
                        print(f"U-Net: {processed}/{len(rows)} images in {time.perf_counter() - start:.1f}s", flush=True)
                    batch_images, batch_rows, batch_masks = [], [], []

            if batch_images:
                t0 = time.perf_counter()
                preds = model.predict(np.stack(batch_images, axis=0), verbose=0)
                infer_time = (time.perf_counter() - t0) / len(batch_images)
                for pred_raw, item_row, gt in zip(preds, batch_rows, batch_masks):
                    pred = np.squeeze(pred_raw) >= args.threshold
                    item = metrics(pred, gt)
                    item["model"] = "U-Net"
                    item["image"] = item_row["image"]
                    item["inference_time_s"] = infer_time
                    writer.writerow(item)
                    values.append(item)
                out_fh.flush()

    if not values and per_image_path.exists():
        with open(per_image_path, newline="", encoding="utf-8") as fh:
            values = list(csv.DictReader(fh))

    summary = {"model": "U-Net", "n": len(values), "elapsed_s": time.perf_counter() - start}
    for key in ["iou", "dice", "precision", "recall", "pixel_accuracy", "inference_time_s"]:
        arr = np.array([float(v[key]) for v in values], dtype=float)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std())

    summary_path = out_dir / "cross_dataset_unet_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as summary_fh:
        writer = csv.DictWriter(summary_fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
