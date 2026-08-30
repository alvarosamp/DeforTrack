"""Evaluate YOLO segmentation checkpoints on the 892-image DeforTrack test set."""

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


EPS = 1e-7


def yolo_mask(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask.astype(bool)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        coords = np.asarray(values[1:], dtype=np.float32)
        if len(coords) % 2:
            continue
        points = coords.reshape(-1, 2)
        points[:, 0] = np.clip(np.rint(points[:, 0] * (width - 1)), 0, width - 1)
        points[:, 1] = np.clip(np.rint(points[:, 1] * (height - 1)), 0, height - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask.astype(bool)


def boundary(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    source = mask.astype(np.uint8)
    if not source.any():
        return np.zeros_like(source, dtype=bool)
    eroded = cv2.erode(source, np.ones((3, 3), dtype=np.uint8), iterations=thickness)
    return (source - eroded).astype(bool)


def dilate(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    union = tp + fp + fn
    dice_den = 2 * tp + fp + fn
    pb = boundary(pred)
    gb = boundary(gt)
    pband = dilate(pb)
    gband = dilate(gb)
    b_union = np.logical_or(pband, gband).sum()
    b_inter = np.logical_and(pband, gband).sum()
    bp = np.logical_and(pb, gband).sum() / (pb.sum() + EPS) if pb.any() else (1.0 if not gb.any() else 0.0)
    br = np.logical_and(gb, pband).sum() / (gb.sum() + EPS) if gb.any() else (1.0 if not pb.any() else 0.0)
    return {
        "iou": tp / (union + EPS) if union else 1.0,
        "dice_f1": 2 * tp / (dice_den + EPS) if dice_den else 1.0,
        "precision": tp / (tp + fp + EPS) if tp + fp else (1.0 if not gt.any() else 0.0),
        "recall": tp / (tp + fn + EPS) if tp + fn else (1.0 if not pred.any() else 0.0),
        "boundary_iou": b_inter / (b_union + EPS) if b_union else (1.0 if np.array_equal(pred, gt) else 0.0),
        "boundary_f1": 2 * bp * br / (bp + br + EPS) if bp + br else 0.0,
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + EPS),
    }


def prediction_mask(result, shape: tuple[int, int]) -> np.ndarray:
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return np.zeros(shape, dtype=bool)
    merged = result.masks.data.detach().cpu().numpy().max(axis=0)
    if merged.shape != shape:
        merged = cv2.resize(merged.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return merged > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = json.loads(Path(args.models_json).read_text(encoding="utf-8"))
    summary_path = out_dir / "yolo_test_summary.csv"
    per_image_path = out_dir / "yolo_test_per_image.csv"
    completed = set()
    if args.resume and summary_path.exists():
        completed = {r["model"] for r in csv.DictReader(summary_path.open(encoding="utf-8"))}

    fields = ["model", "image", "iou", "dice_f1", "precision", "recall", "boundary_iou", "boundary_f1", "pixel_accuracy", "inference_time_s"]
    append_per = args.resume and per_image_path.exists()
    with per_image_path.open("a" if append_per else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append_per:
            writer.writeheader()
        for model_name, model_path in models.items():
            if model_name in completed:
                print(f"Skipping complete model: {model_name}", flush=True)
                continue
            print(f"Loading {model_name}", flush=True)
            model = YOLO(model_path)
            rows = []
            started = time.perf_counter()
            for start in range(0, len(images), args.batch_size):
                paths = images[start:start + args.batch_size]
                t0 = time.perf_counter()
                predictions = model.predict([str(p) for p in paths], imgsz=args.imgsz, conf=args.conf, device=args.device, batch=args.batch_size, verbose=False)
                infer_time = (time.perf_counter() - t0) / len(paths)
                for path, result in zip(paths, predictions):
                    image = cv2.imread(str(path))
                    gt = yolo_mask(labels / f"{path.stem}.txt", image.shape[0], image.shape[1])
                    row = metrics(prediction_mask(result, gt.shape), gt)
                    row.update(model=model_name, image=path.name, inference_time_s=infer_time)
                    rows.append(row)
                    writer.writerow(row)
                handle.flush()
                if start + len(paths) == len(images) or (start + len(paths)) % 100 < args.batch_size:
                    print(f"{model_name}: {start + len(paths)}/{len(images)}", flush=True)

            val = model.val(data=args.data_yaml, split="val", imgsz=args.imgsz, batch=args.batch_size, device=args.device, verbose=False, plots=False, save_json=False)
            summary = {"model": model_name, "n": len(rows), "elapsed_s": time.perf_counter() - started, "map_50": float(val.seg.map50), "map_50_95": float(val.seg.map)}
            for key in fields[2:]:
                values = np.asarray([row[key] for row in rows], dtype=float)
                summary[f"{key}_mean"] = float(values.mean())
                summary[f"{key}_std"] = float(values.std(ddof=1))
                summary[f"{key}_ci95_halfwidth"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
            write_header = not summary_path.exists()
            with summary_path.open("a", newline="", encoding="utf-8") as summary_handle:
                summary_writer = csv.DictWriter(summary_handle, fieldnames=summary.keys())
                if write_header:
                    summary_writer.writeheader()
                summary_writer.writerow(summary)
            print(json.dumps(summary, indent=2), flush=True)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
