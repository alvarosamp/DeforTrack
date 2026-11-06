import argparse
import csv
import io
import json
import os
import random
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

torch.set_num_threads(1)


def read_mask(mask_zip, mask_name):
    zip_name = f"Forest Segmented/Forest Segmented/masks/{mask_name}"
    with mask_zip.open(zip_name) as fh:
        mask = Image.open(io.BytesIO(fh.read())).convert("L")
    return np.array(mask) > 127


def resolve_image(image_dir, image_zip, image_name, cache_dir):
    image_path = image_dir / image_name
    if image_path.exists():
        return image_path

    cached_path = cache_dir / image_name
    if not cached_path.exists():
        zip_name = f"Forest Segmented/Forest Segmented/images/{image_name}"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with image_zip.open(zip_name) as fh:
            cached_path.write_bytes(fh.read())
    return cached_path


def predict_binary_mask(model, image_path, shape, imgsz, conf):
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        device="cpu",
    )[0]

    if result.masks is None or result.masks.data is None:
        return np.zeros(shape, dtype=bool)

    masks = result.masks.data.cpu().numpy()
    if masks.size == 0:
        return np.zeros(shape, dtype=bool)

    merged = masks.max(axis=0)
    resized = cv2.resize(
        merged.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def result_to_binary_mask(result, shape):
    if result.masks is None or result.masks.data is None:
        return np.zeros(shape, dtype=bool)

    masks = result.masks.data.cpu().numpy()
    if masks.size == 0:
        return np.zeros(shape, dtype=bool)

    merged = masks.max(axis=0)
    resized = cv2.resize(
        merged.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


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
    parser.add_argument("--mask-zip", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    image_dir = dataset_root / "images"
    metadata_path = dataset_root / "meta_data.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = out_dir / "_image_cache"

    with open(args.models_json, encoding="utf-8") as fh:
        models = json.load(fh)

    with open(metadata_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.sample_size:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, min(args.sample_size, len(rows)))
    if args.limit:
        rows = rows[: args.limit]

    summary_rows = []
    summary_path = out_dir / "cross_dataset_yolo_summary.csv"
    per_image_path = out_dir / "cross_dataset_yolo_per_image.csv"
    completed_models = set()
    completed_images_by_model = {}
    if args.resume and summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as fh:
            completed_models = {row["model"] for row in csv.DictReader(fh)}
    if args.resume and per_image_path.exists():
        with open(per_image_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                completed_images_by_model.setdefault(row["model"], set()).add(row["image"])

    per_image_exists = args.resume and per_image_path.exists()
    with open(per_image_path, "a" if per_image_exists else "w", newline="", encoding="utf-8") as per_fh:
        writer = csv.DictWriter(
            per_fh,
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
        if not per_image_exists:
            writer.writeheader()

        with zipfile.ZipFile(args.mask_zip) as mask_zip:
            for model_name, model_path in models.items():
                if model_name in completed_models:
                    print(f"Skipping completed model: {model_name}", flush=True)
                    continue

                model = YOLO(model_path)
                values = []
                done_images = completed_images_by_model.get(model_name, set())
                start_model = time.perf_counter()
                pending = [row for row in rows if row["image"] not in done_images]
                for start_idx in range(0, len(pending), args.batch_size):
                    batch_rows = pending[start_idx:start_idx + args.batch_size]
                    image_paths = []
                    gt_masks = []
                    for row in batch_rows:
                        image_paths.append(resolve_image(image_dir, mask_zip, row["image"], image_cache_dir))
                        gt_masks.append(read_mask(mask_zip, row["mask"]))

                    t0 = time.perf_counter()
                    results = model.predict(
                        source=[str(p) for p in image_paths],
                        imgsz=args.imgsz,
                        conf=args.conf,
                        verbose=False,
                        device=args.device,
                        batch=args.batch_size,
                    )
                    infer_time = (time.perf_counter() - t0) / len(batch_rows)

                    for row, gt, result in zip(batch_rows, gt_masks, results):
                        pred = result_to_binary_mask(result, gt.shape)
                        item = metrics(pred, gt)
                        item["inference_time_s"] = infer_time
                        item["model"] = model_name
                        item["image"] = row["image"]
                        writer.writerow(item)
                        values.append(item)

                    per_fh.flush()
                    completed_count = len(done_images) + len(values)
                    if args.progress_every and completed_count % args.progress_every < args.batch_size:
                        elapsed = time.perf_counter() - start_model
                        print(f"{model_name}: {completed_count}/{len(rows)} images in {elapsed:.1f}s", flush=True)

                elapsed = time.perf_counter() - start_model
                all_model_values = values
                if done_images:
                    with open(per_image_path, newline="", encoding="utf-8") as fh:
                        all_model_values = [
                            row for row in csv.DictReader(fh)
                            if row["model"] == model_name
                        ]

                summary = {"model": model_name, "n": len(all_model_values), "elapsed_s": elapsed}
                for key in ["iou", "dice", "precision", "recall", "pixel_accuracy", "inference_time_s"]:
                    arr = np.array([float(v[key]) for v in all_model_values], dtype=float)
                    summary[f"{key}_mean"] = float(arr.mean())
                    summary[f"{key}_std"] = float(arr.std())
                summary_rows.append(summary)
                print(json.dumps(summary, indent=2), flush=True)

                fieldnames = list(summary.keys())
                write_header = not summary_path.exists()
                with open(summary_path, "a", newline="", encoding="utf-8") as summary_fh:
                    summary_writer = csv.DictWriter(summary_fh, fieldnames=fieldnames)
                    if write_header:
                        summary_writer.writeheader()
                    summary_writer.writerow(summary)


if __name__ == "__main__":
    main()
