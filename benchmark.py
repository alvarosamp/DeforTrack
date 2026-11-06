import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
COUNTS = {"train": 12861, "valid": 833, "test": 892, "ForestAerial": 5022, "AmazonAtlantic": 599}
THRESHOLDS = {"yolov8": 0.25, "yolov11": 0.25, "mask-rcnn": 0.30, "u-net": 0.50}
FAMILIES = {"yolov8": "ultralytics", "yolov11": "ultralytics", "mask-rcnn": "detectron2", "u-net": "keras"}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def check_file(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def rows_for(dataset):
    if dataset in ("ForestAerial", "AmazonAtlantic"):
        rows = read_csv(HERE / "data" / f"{dataset}.csv")
    else:
        manifest = HERE / "data/partition_manifest.csv"
        protocol = json.loads((HERE / "data/protocol.json").read_text(encoding="utf-8"))
        check_file(manifest, protocol["manifest_sha256"])
        rows = [row for row in read_csv(manifest) if row["split"] == dataset]
    if len(rows) != COUNTS[dataset]:
        raise ValueError(f"Unexpected number of manifest rows for {dataset}")
    return rows


def checked_pairs(dataset, root, rows):
    internal = dataset in ("train", "valid", "test")
    folder = root / dataset if internal else root
    field = "label" if internal else "mask"
    for row in rows:
        image = folder / "images" / row["image"]
        mask = folder / ("labels" if internal else "masks") / row[field]
        check_file(image, row["image_sha256"])
        check_file(mask, row[field + "_sha256"])
        yield row, image, mask


def model_record(name, weights):
    rows = [row for row in read_csv(HERE / "models/checkpoints.csv") if row["model"] == name]
    if len(rows) != 1:
        raise ValueError(f"Unknown checkpoint: {name}")
    record = rows[0]
    check_file(weights, record["sha256"])
    return record


def load_predictor(record, weights):
    from run_benchmark import predictor

    if record["architecture"] != "mask-rcnn":
        return predictor({"family": FAMILIES[record["architecture"]]}, {"best_checkpoint": str(weights)})

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(str(HERE / record["config"]))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    model = DefaultPredictor(cfg)

    def infer(image, score):
        instances = model(image)["instances"].to("cpu")
        keep = instances.scores.numpy() >= score
        return instances.pred_masks.numpy()[keep], instances.scores.numpy()[keep].tolist()

    return infer


def evaluate(args):
    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, str(HERE / "training"))
    from yolov8_yolov11.evaluate import metrics, yolo_mask

    record = model_record(args.model, args.weights)
    rows = rows_for(args.dataset)
    if not torch.cuda.is_available():
        raise RuntimeError("The evaluation uses a CUDA GPU")
    infer = load_predictor(record, args.weights)
    threshold = THRESHOLDS[record["architecture"]]
    score_ap = args.dataset == "test" and record["architecture"] != "u-net"
    if score_ap:
        from run_benchmark import test_ground_truth
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    ledger, detections = [], []
    for index, (row, path, mask_path) in enumerate(checked_pairs(args.dataset, args.data, rows), 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        if args.dataset == "test":
            reference = yolo_mask(mask_path, *image.shape[:2])
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != image.shape[:2]:
                raise ValueError(f"Invalid reference mask: {mask_path}")
            reference = mask > 127
        masks, scores = infer(image, 0.001)
        if len(masks) != len(scores):
            raise ValueError(f"Mask/score count mismatch: {path}")
        prediction = np.zeros(image.shape[:2], dtype=bool)
        for mask, score in zip(masks, scores):
            if mask.shape != image.shape[:2] or not 0 <= score <= 1:
                raise ValueError(f"Invalid model output: {path}")
            if score >= threshold:
                prediction |= mask
            if score_ap:
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode()
                detections.append(dict(image_id=index, category_id=1, segmentation=rle, score=float(score)))
        ledger.append(dict(model=args.model, image=row["image"], **{
            key: float(value) for key, value in metrics(prediction, reference).items()
        }))

    summary = dict(model=args.model, dataset=args.dataset, n=len(ledger), weights_sha256=record["sha256"])
    for name in list(ledger[0])[2:]:
        values = np.asarray([row[name] for row in ledger])
        sd = float(values.std(ddof=1))
        summary.update({name + "_mean": float(values.mean()), name + "_std": sd,
                        name + "_ci95_halfwidth": 1.96 * sd / len(values)**0.5})
    if score_ap:
        gt = COCO()
        gt.dataset = test_ground_truth(rows, args.data)
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = dict(images=gt.dataset["images"], categories=gt.dataset["categories"], annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "segm")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        summary["common_ap"] = dict(zip(("AP", "AP50", "AP75"), map(float, evaluator.stats[:3])))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"{args.dataset}_per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    (args.output / f"{args.dataset}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-data")
    verify.add_argument("dataset", choices=COUNTS)
    verify.add_argument("data", type=Path)
    score = commands.add_parser("evaluate")
    score.add_argument("dataset", choices=("test", "ForestAerial", "AmazonAtlantic"))
    score.add_argument("data", type=Path)
    score.add_argument("--model", required=True)
    score.add_argument("--weights", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify-data":
        count = sum(1 for _ in checked_pairs(args.dataset, args.data, rows_for(args.dataset)))
        print(f"{args.dataset}: {count} image/reference pairs verified")
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
