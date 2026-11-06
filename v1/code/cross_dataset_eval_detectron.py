import argparse
import csv
import gc
import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

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


def read_image(dataset_root, dataset_zip, image_name, cache_dir):
    image_path = dataset_root / "images" / image_name
    if image_path.exists():
        return str(image_path)

    cached_path = cache_dir / image_name
    if not cached_path.exists():
        zip_name = f"Forest Segmented/Forest Segmented/images/{image_name}"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(dataset_zip.read(zip_name))
    return str(cached_path)


def read_mask(dataset_zip, mask_name):
    zip_name = f"Forest Segmented/Forest Segmented/masks/{mask_name}"
    mask = Image.open(io.BytesIO(dataset_zip.read(zip_name))).convert("L")
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


def build_predictor(model_name, weights, device, score_thresh, num_classes, image_size, max_detections, proposal_topk):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(ARCH_TO_CONFIG[model_name]))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = max(proposal_topk * 5, proposal_topk)
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = proposal_topk
    cfg.TEST.DETECTIONS_PER_IMAGE = max_detections
    cfg.INPUT.MIN_SIZE_TEST = image_size
    cfg.INPUT.MAX_SIZE_TEST = image_size
    return DefaultPredictor(cfg)


def result_to_mask(outputs, shape, accepted_classes):
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0 or not instances.has("pred_masks"):
        return np.zeros(shape, dtype=bool)

    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros(len(masks))
    keep = np.isin(classes, accepted_classes)
    if not keep.any():
        return np.zeros(shape, dtype=bool)

    merged = masks[keep].max(axis=0)
    if merged.shape != shape:
        merged = cv2.resize(merged.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return merged.astype(bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--accepted-classes", default="0,1")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--proposal-topk", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = out_dir / "_image_cache"

    with open(dataset_root / "meta_data.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    with open(args.models_json, encoding="utf-8") as fh:
        models = json.load(fh)

    accepted_classes = [int(x) for x in args.accepted_classes.split(",") if x.strip()]
    per_image_path = out_dir / "cross_dataset_detectron_per_image.csv"
    summary_path = out_dir / "cross_dataset_detectron_summary.csv"

    completed_models = set()
    completed_images_by_model = {}
    if args.resume and summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as fh:
            completed_models = {row["model"] for row in csv.DictReader(fh)}
    if args.resume and per_image_path.exists():
        with open(per_image_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                completed_images_by_model.setdefault(row["model"], set()).add(row["image"])

    per_exists = args.resume and per_image_path.exists()
    with open(per_image_path, "a" if per_exists else "w", newline="", encoding="utf-8") as per_fh:
        writer = csv.DictWriter(
            per_fh,
            fieldnames=["model", "image", "iou", "dice", "precision", "recall", "pixel_accuracy", "inference_time_s"],
        )
        if not per_exists:
            writer.writeheader()

        with zipfile.ZipFile(args.dataset_zip) as dataset_zip:
            for model_name, weights in models.items():
                if model_name in completed_models:
                    print(f"Skipping completed model: {model_name}", flush=True)
                    continue

                predictor = build_predictor(
                    model_name,
                    weights,
                    args.device,
                    args.score_thresh,
                    args.num_classes,
                    args.image_size,
                    args.max_detections,
                    args.proposal_topk,
                )
                done_images = completed_images_by_model.get(model_name, set())
                values = []
                start_model = time.perf_counter()

                for row in rows:
                    if row["image"] in done_images:
                        continue
                    image_path = read_image(dataset_root, dataset_zip, row["image"], image_cache_dir)
                    image = cv2.imread(image_path)
                    gt = read_mask(dataset_zip, row["mask"])

                    t0 = time.perf_counter()
                    outputs = predictor(image)
                    infer_time = time.perf_counter() - t0

                    pred = result_to_mask(outputs, gt.shape, accepted_classes)
                    item = metrics(pred, gt)
                    item["model"] = model_name
                    item["image"] = row["image"]
                    item["inference_time_s"] = infer_time
                    writer.writerow(item)
                    per_fh.flush()
                    values.append(item)

                    completed_count = len(done_images) + len(values)
                    if args.progress_every and completed_count % args.progress_every == 0:
                        elapsed = time.perf_counter() - start_model
                        print(f"{model_name}: {completed_count}/{len(rows)} images in {elapsed:.1f}s", flush=True)

                elapsed = time.perf_counter() - start_model
                with open(per_image_path, newline="", encoding="utf-8") as fh:
                    all_model_values = [r for r in csv.DictReader(fh) if r["model"] == model_name]

                summary = {"model": model_name, "n": len(all_model_values), "elapsed_s": elapsed}
                for key in ["iou", "dice", "precision", "recall", "pixel_accuracy", "inference_time_s"]:
                    arr = np.array([float(v[key]) for v in all_model_values], dtype=float)
                    summary[f"{key}_mean"] = float(arr.mean())
                    summary[f"{key}_std"] = float(arr.std())

                write_header = not summary_path.exists()
                with open(summary_path, "a", newline="", encoding="utf-8") as summary_fh:
                    summary_writer = csv.DictWriter(summary_fh, fieldnames=list(summary.keys()))
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
