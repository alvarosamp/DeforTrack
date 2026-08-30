"""Evaluate all supplied Mask R-CNN checkpoints on a YOLO-segmentation split.

The script preserves the binary forest/non-forest protocol used in the article:
all predicted instances from classes 0 and 1 are merged into one forest mask.
It writes resumable per-image and per-model CSV files.
"""

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


ARCH_TO_CONFIG = {
    "mask_rcnn_R_101_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    "mask_rcnn_R_101_DC5_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "mask_rcnn_R_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "mask_rcnn_R_50_C4_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
    "mask_rcnn_R_50_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
    "mask_rcnn_R_50_DC5_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
}

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


def boundary(mask: np.ndarray, thickness: int) -> np.ndarray:
    source = mask.astype(np.uint8)
    if not source.any():
        return np.zeros_like(source, dtype=bool)
    eroded = cv2.erode(source, np.ones((3, 3), dtype=np.uint8), iterations=thickness)
    return (source - eroded).astype(bool)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray, thickness: int) -> dict:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn

    pred_boundary = boundary(pred, thickness)
    gt_boundary = boundary(gt, thickness)
    pred_band = dilate(pred_boundary, thickness)
    gt_band = dilate(gt_boundary, thickness)
    boundary_union = np.logical_or(pred_band, gt_band).sum()
    boundary_intersection = np.logical_and(pred_band, gt_band).sum()
    pred_count = pred_boundary.sum()
    gt_count = gt_boundary.sum()
    b_precision = np.logical_and(pred_boundary, gt_band).sum() / (pred_count + EPS) if pred_count else (1.0 if not gt_count else 0.0)
    b_recall = np.logical_and(gt_boundary, pred_band).sum() / (gt_count + EPS) if gt_count else (1.0 if not pred_count else 0.0)

    return {
        "iou": tp / (union + EPS) if union else 1.0,
        "dice_f1": (2 * tp) / (dice_denominator + EPS) if dice_denominator else 1.0,
        "precision": tp / (tp + fp + EPS) if tp + fp else (1.0 if not gt.any() else 0.0),
        "recall": tp / (tp + fn + EPS) if tp + fn else (1.0 if not pred.any() else 0.0),
        "boundary_iou": boundary_intersection / (boundary_union + EPS) if boundary_union else (1.0 if np.array_equal(pred, gt) else 0.0),
        "boundary_f1": 2 * b_precision * b_recall / (b_precision + b_recall + EPS) if b_precision + b_recall else 0.0,
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + EPS),
    }


def build_predictor(model_name: str, weights: str, device: str, score_thresh: float, image_size: int) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(ARCH_TO_CONFIG[model_name]))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 100
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 20
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.INPUT.MIN_SIZE_TEST = image_size
    cfg.INPUT.MAX_SIZE_TEST = image_size
    return DefaultPredictor(cfg)


def prediction_mask(outputs: dict, shape: tuple[int, int]) -> np.ndarray:
    instances = outputs["instances"].to("cpu")
    if not len(instances) or not instances.has("pred_masks"):
        return np.zeros(shape, dtype=bool)
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros(len(instances), dtype=int)
    keep = np.isin(classes, [0, 1])
    if not keep.any():
        return np.zeros(shape, dtype=bool)
    merged = instances.pred_masks.numpy()[keep].any(axis=0)
    if merged.shape != shape:
        merged = cv2.resize(merged.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--boundary-thickness", type=int, default=3)
    parser.add_argument("--model", default="", help="Evaluate only one model key from models.json.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    image_paths = sorted(path for path in Path(args.images).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        image_paths = image_paths[: args.limit]
    labels_dir = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / "maskrcnn_test_per_image.csv"
    summary_path = out_dir / "maskrcnn_test_summary.csv"
    models = json.loads(Path(args.models_json).read_text(encoding="utf-8"))
    if args.model:
        if args.model not in models:
            raise ValueError(f"Model not found in models.json: {args.model}")
        models = {args.model: models[args.model]}

    completed = set()
    if args.resume and summary_path.exists():
        completed = {row["model"] for row in csv.DictReader(summary_path.open(encoding="utf-8"))}

    has_per_file = args.resume and per_path.exists()
    with per_path.open("a" if has_per_file else "w", newline="", encoding="utf-8") as per_file:
        writer = csv.DictWriter(per_file, fieldnames=["model", "image", "iou", "dice_f1", "precision", "recall", "boundary_iou", "boundary_f1", "pixel_accuracy", "inference_time_s"])
        if not has_per_file:
            writer.writeheader()
        for model_name, weights in models.items():
            if model_name in completed:
                print(f"Skipping complete model: {model_name}", flush=True)
                continue
            print(f"Loading {model_name} on {args.device}", flush=True)
            predictor = build_predictor(model_name, weights, args.device, args.score_thresh, args.image_size)
            started = time.perf_counter()
            rows = []
            for index, image_path in enumerate(image_paths, start=1):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                gt = yolo_mask(labels_dir / f"{image_path.stem}.txt", image.shape[0], image.shape[1])
                t0 = time.perf_counter()
                pred = prediction_mask(predictor(image), gt.shape)
                item = segmentation_metrics(pred, gt, args.boundary_thickness)
                item.update(model=model_name, image=image_path.name, inference_time_s=time.perf_counter() - t0)
                writer.writerow(item)
                per_file.flush()
                rows.append(item)
                if index % 50 == 0:
                    print(f"{model_name}: {index}/{len(image_paths)}", flush=True)

            summary = {"model": model_name, "n": len(rows), "elapsed_s": time.perf_counter() - started}
            for key in ["iou", "dice_f1", "precision", "recall", "boundary_iou", "boundary_f1", "pixel_accuracy", "inference_time_s"]:
                values = np.array([row[key] for row in rows], dtype=float)
                summary[f"{key}_mean"] = values.mean()
                summary[f"{key}_std"] = values.std(ddof=1)
            write_header = not summary_path.exists()
            with summary_path.open("a", newline="", encoding="utf-8") as summary_file:
                summary_writer = csv.DictWriter(summary_file, fieldnames=summary.keys())
                if write_header:
                    summary_writer.writeheader()
                summary_writer.writerow(summary)
            print(json.dumps(summary, indent=2), flush=True)
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
