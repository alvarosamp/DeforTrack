"""Compute COCO mask AP for Mask R-CNN checkpoints on a YOLO-segmentation split."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ARCH_TO_CONFIG = {
    "mask_rcnn_R_101_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    "mask_rcnn_R_101_DC5_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "mask_rcnn_R_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "mask_rcnn_R_50_C4_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
    "mask_rcnn_R_50_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
    "mask_rcnn_R_50_DC5_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
}


def rle_for(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def bbox_for(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def yolo_instances(label: Path, height: int, width: int) -> list[np.ndarray]:
    instances = []
    if not label.exists():
        return instances
    for line in label.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        xy = np.asarray(values[1:], dtype=np.float32)
        if len(xy) % 2:
            continue
        points = xy.reshape(-1, 2)
        points[:, 0] = np.clip(np.rint(points[:, 0] * (width - 1)), 0, width - 1)
        points[:, 1] = np.clip(np.rint(points[:, 1] * (height - 1)), 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        if mask.any():
            instances.append(mask.astype(bool))
    return instances


def predictor(model_name: str, weights: str, score: float, image_size: int) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(ARCH_TO_CONFIG[model_name]))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 100
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 20
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.INPUT.MIN_SIZE_TEST = image_size
    cfg.INPUT.MAX_SIZE_TEST = image_size
    return DefaultPredictor(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--score-thresh", type=float, default=0.30)
    args = parser.parse_args()

    images = sorted(path for path in Path(args.images).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = Path(args.labels)
    gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "forest"}]}
    image_info = []
    annotation_id = 1
    for image_id, path in enumerate(images, start=1):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        gt["images"].append({"id": image_id, "width": width, "height": height, "file_name": path.name})
        image_info.append((image_id, path, image, height, width))
        for instance in yolo_instances(labels / f"{path.stem}.txt", height, width):
            gt["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle_for(instance),
                "area": float(instance.sum()),
                "bbox": bbox_for(instance),
                "iscrowd": 0,
            })
            annotation_id += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    gt_path = output.with_name("maskrcnn_892_coco_ground_truth.json")
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    coco_gt = COCO(str(gt_path))
    models = json.loads(Path(args.models_json).read_text(encoding="utf-8"))
    summaries = []

    for model_name, weights in models.items():
        print(f"Loading {model_name}", flush=True)
        model = predictor(model_name, weights, args.score_thresh, args.image_size)
        predictions = []
        for idx, (image_id, _, image, height, width) in enumerate(image_info, start=1):
            result = model(image)["instances"].to("cpu")
            if len(result) and result.has("pred_masks"):
                classes = result.pred_classes.numpy()
                masks = result.pred_masks.numpy()
                scores = result.scores.numpy()
                for instance, score, category in zip(masks, scores, classes):
                    if category not in (0, 1) or not instance.any():
                        continue
                    if instance.shape != (height, width):
                        instance = cv2.resize(instance.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    predictions.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": rle_for(instance),
                        "score": float(score),
                    })
            if idx % 50 == 0:
                print(f"{model_name}: {idx}/{len(image_info)}", flush=True)

        prediction_path = output.with_name(f"{model_name}_coco_predictions.json")
        prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
        coco_dt = coco_gt.loadRes(str(prediction_path)) if predictions else COCO()
        evaluator = COCOeval(coco_gt, coco_dt, "segm")
        evaluator.params.imgIds = [item[0] for item in image_info]
        evaluator.params.catIds = [1]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        summary = {"model": model_name, "images": len(image_info), "map_50_95": float(evaluator.stats[0]), "map_50": float(evaluator.stats[1]), "predicted_instances": len(predictions)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
