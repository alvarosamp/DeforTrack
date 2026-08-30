"""Evaluate the supplied U-Net on the 892-image DeforTrack split."""

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import keras
import numpy as np
from PIL import Image


EPS = 1e-7


def yolo_mask(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask.astype(bool)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        xy = np.asarray(values[1:], dtype=np.float32)
        if len(xy) % 2:
            continue
        points = xy.reshape(-1, 2)
        points[:, 0] = np.clip(np.rint(points[:, 0] * (width - 1)), 0, width - 1)
        points[:, 1] = np.clip(np.rint(points[:, 1] * (height - 1)), 0, height - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask.astype(bool)


def boundary(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    binary = mask.astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    return (binary - cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=thickness)).astype(bool)


def dilate(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    pred_boundary = boundary(pred)
    gt_boundary = boundary(gt)
    pred_band = dilate(pred_boundary)
    gt_band = dilate(gt_boundary)
    b_union = np.logical_or(pred_band, gt_band).sum()
    b_intersection = np.logical_and(pred_band, gt_band).sum()
    b_precision = np.logical_and(pred_boundary, gt_band).sum() / (pred_boundary.sum() + EPS) if pred_boundary.any() else (1.0 if not gt_boundary.any() else 0.0)
    b_recall = np.logical_and(gt_boundary, pred_band).sum() / (gt_boundary.sum() + EPS) if gt_boundary.any() else (1.0 if not pred_boundary.any() else 0.0)
    return {
        "iou": tp / (union + EPS) if union else 1.0,
        "dice_f1": 2 * tp / (dice_denominator + EPS) if dice_denominator else 1.0,
        "precision": tp / (tp + fp + EPS) if tp + fp else (1.0 if not gt.any() else 0.0),
        "recall": tp / (tp + fn + EPS) if tp + fn else (1.0 if not pred.any() else 0.0),
        "boundary_iou": b_intersection / (b_union + EPS) if b_union else (1.0 if np.array_equal(pred, gt) else 0.0),
        "boundary_f1": 2 * b_precision * b_recall / (b_precision + b_recall + EPS) if b_precision + b_recall else 0.0,
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + EPS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    image_paths = sorted(path for path in Path(args.images).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    label_dir = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = keras.saving.load_model(args.model, compile=False)
    rows = []
    started = time.perf_counter()

    for start in range(0, len(image_paths), args.batch_size):
        paths = image_paths[start : start + args.batch_size]
        inputs, targets = [], []
        for path in paths:
            image = Image.open(path).convert("RGB").resize((256, 256))
            inputs.append(np.asarray(image, dtype=np.float32) / 255.0)
            targets.append(yolo_mask(label_dir / f"{path.stem}.txt", 256, 256))
        t0 = time.perf_counter()
        predictions = model.predict(np.stack(inputs), verbose=0)
        per_image_time = (time.perf_counter() - t0) / len(paths)
        for path, prediction, target in zip(paths, predictions, targets):
            pred = np.squeeze(prediction) >= args.threshold
            row = metrics(pred.astype(bool), target)
            row.update(model="U-Net", image=path.name, inference_time_s=per_image_time)
            rows.append(row)
        if (start + len(paths)) % 100 < args.batch_size or start + len(paths) == len(image_paths):
            print(f"U-Net: {start + len(paths)}/{len(image_paths)}", flush=True)

    fields = ["model", "image", "iou", "dice_f1", "precision", "recall", "boundary_iou", "boundary_f1", "pixel_accuracy", "inference_time_s"]
    with (out_dir / "unet_892_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"model": "U-Net", "n": len(rows), "elapsed_s": time.perf_counter() - started}
    for key in fields[2:]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1))
        summary[f"{key}_ci95_halfwidth"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
    with (out_dir / "unet_892_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
