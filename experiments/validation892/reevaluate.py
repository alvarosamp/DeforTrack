"""Reevaluate released checkpoints without training or checkpoint selection."""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
MANIFESTS = EXPERIMENT / "data"
BASELINE = "e14eecf7e8c253dfb1182fc31a916587148f0f3bd66ee21337d5c1b05a090b3c"
KEYS = ("iou", "dice_f1", "precision", "recall", "boundary_iou", "boundary_f1", "pixel_accuracy")
FPN = "mask_rcnn_R_101_FPN_3x"
GRID = [i / 100 for i in range(5, 100, 5)]


def csv_read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize(rows, model, dataset):
    result = dict(model=model, dataset=dataset, n=len(rows))
    for key in KEYS:
        values = [float(r[key]) for r in rows]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f"Invalid {model} {dataset} metric: {key}")
        sd = statistics.stdev(values)
        result.update({key + "_mean": statistics.mean(values), key + "_std": sd,
                       key + "_ci95_halfwidth": 1.96 * sd / math.sqrt(len(values))})
    return result


def ledger(path, model, names):
    rows = csv_read(path)
    if len(rows) != len(names) or {r["image"] for r in rows} != names or {r["model"] for r in rows} != {model}:
        raise ValueError(f"Incomplete or incompatible ledger: {path}")
    return rows


def aggregate(out, models, old_rows, protocol):
    screens = (("ForestAerial", "forest_aerial_overlap.csv", "forest_aerial_external"),
               ("AmazonAtlantic", "amazon_atlantic_overlap.csv", "amazon_atlantic_external"))
    targets = {}
    for name, filename, count_key in screens:
        original = {r["image"]: (r["image_sha256"], r["mask_sha256"])
                    for r in csv_read(REPO / "data" / (name + ".csv"))}
        old_names = set(original)
        if sha(MANIFESTS / filename) != protocol["external_screen_sha256"][filename]:
            raise ValueError(f"External screening hash mismatch: {name}")
        screened = [r for r in csv_read(MANIFESTS / filename) if r["included_in_screened_subset"].lower() == "true"]
        retained = {r["image"] for r in screened}
        if len(retained) != protocol[count_key] or not retained <= old_names:
            raise ValueError(f"Invalid retained external set: {name}")
        current = read(MANIFESTS / (name + "_content_manifest.json"))
        if {r["image"] for r in current} != retained or any(
            original[r["image"]] != (r["image_sha256"], r["mask_sha256"]) for r in current
        ):
            raise ValueError(f"External bytes do not match published evaluation: {name}")
        targets[name] = old_names, retained
    test = {r["image"] for r in old_rows if r["split"] == "test"}
    summaries, internal = [], []
    for item in models:
        model = item["model"]
        folder = REPO / "models" / item["architecture"] / model
        for directory in ("internal", "external", "profile"):
            if read(folder / directory / "summary.json")["weights_sha256"] != item["sha256"]:
                raise ValueError(f"Published summary hash mismatch: {model}/{directory}")
        result = summarize(ledger(folder / "internal/test_per_image.csv", model, test), model, "test")
        internal.append(result)
        for name, (old_names, retained) in targets.items():
            rows = [r for r in ledger(folder / "external" / (name + "_per_image.csv"), model, old_names)
                    if r["image"] in retained]
            summaries.append(summarize(rows, model, name))
    ranking = []
    for name in targets:
        ordered = sorted((r for r in summaries if r["dataset"] == name), key=lambda r: -r["iou_mean"])
        ranking.extend(dict(rank=i, **r) for i, r in enumerate(ordered, 1))
    csv_write(out / "external_rankings.csv", ranking)
    csv_write(out / "internal_test_region_unchanged.csv", internal)
    print("Reaggregated external and internal region metrics for all 17 frozen checkpoints", flush=True)


def compare_validation(out, models, old_rows, rows):
    archived = EXPERIMENT / "results"
    old = {r["image"] for r in old_rows if r["split"] == "valid"}
    current = {r["image"] for r in rows if r["split"] == "valid"}
    additions = {r["image"] for r in csv_read(MANIFESTS / "selected_sources.csv")}
    if len(old) != 833 or len(additions) != 59 or old & additions or old | additions != current:
        raise ValueError("Validation partitions do not match the 59 additions")
    summaries = {r["model"]: r for r in csv_read(archived / "validation_summary.csv")}
    comparison = []
    for item in models:
        name = item["model"]
        scores = ledger(archived / "per_image" / (name + "_validation_per_image.csv"), name, current)
        full = summarize(scores, name, "validation892")
        saved = summaries[name]
        if saved["weights_sha256"] != item["sha256"] or saved["selection_images"] != "833":
            raise ValueError(f"Checkpoint provenance changed: {name}")
        for key in KEYS:
            for suffix in ("_mean", "_std", "_ci95_halfwidth"):
                field = key + suffix
                if not math.isclose(full[field], float(saved[field]), rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Validation summary mismatch: {name}/{field}")
        old_iou = statistics.mean(float(r["iou"]) for r in scores if r["image"] in old)
        added_iou = statistics.mean(float(r["iou"]) for r in scores if r["image"] in additions)
        comparison.append(dict(model=name, original_images=833, added_images=59, expanded_images=892,
                               original_mean_iou=old_iou, added_mean_iou=added_iou,
                               expanded_mean_iou=full["iou_mean"],
                               delta_percentage_points=100 * (full["iou_mean"] - old_iou)))
    comparison.sort(key=lambda r: -r["expanded_mean_iou"])
    csv_write(out / "validation_comparison.csv", comparison)
    print("Verified the archived validation region metrics for all 17 frozen checkpoints", flush=True)


def evaluate(data, weights_dir, out, models, rows):
    os.environ.setdefault("KERAS_BACKEND", "torch")
    sys.path[:0] = [str(REPO), str(REPO / "training")]
    import cv2
    import numpy as np
    import torch
    from benchmark import THRESHOLDS, load_predictor
    from mask_rcnn.data import instance_masks
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from yolov8_yolov11.evaluate import metrics, yolo_mask

    if not torch.cuda.is_available():
        raise ValueError("CUDA required for the released evaluation protocol")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    valid = [r for r in rows if r["split"] == "valid"]
    test = [r for r in rows if r["split"] == "test"]
    per_image = out / "per_image"
    per_image.mkdir()
    for r in valid + test:
        for kind, field in (("images", "image"), ("labels", "label")):
            if sha(data / r["split"] / kind / r[field]) != r[field + "_sha256"]:
                raise ValueError(f"Changed evaluation input: {r[field]}")

    def rle(mask):
        return mu.encode(np.asfortranarray(mask, dtype=np.uint8))

    gt = COCO()
    gt.dataset = dict(info={}, images=[], annotations=[], categories=[dict(id=1, name="Forest")])
    for i, row in enumerate(valid, 1):
        gt.dataset["images"].append(dict(id=i, width=640, height=640, file_name=row["image"]))
        label = data / "valid/labels" / row["label"]
        for mask in instance_masks(label.read_text(), 640, 640):
            encoding = rle(mask)
            gt.dataset["annotations"].append(dict(id=len(gt.dataset["annotations"])+1, image_id=i,
                                                   category_id=1, iscrowd=0, segmentation=encoding,
                                                   area=float(mask.sum()), bbox=mu.toBbox(encoding).tolist()))
    gt.createIndex()

    def sample(row, infer, minimum):
        part = data / row["split"]
        image = cv2.imread(str(part / "images" / row["image"]))
        if image is None or image.shape[:2] != (640, 640):
            raise ValueError(f"Wrong image grid: {row['image']}")
        reference = yolo_mask(part / "labels" / row["label"], 640, 640)
        masks, scores = infer(image, minimum)
        return reference, masks, scores

    def fuse(reference, masks, scores):
        union = np.zeros_like(reference)
        inter = np.ones_like(reference) if len(masks) else union.copy()
        weighted = np.zeros(reference.shape, np.float64)
        for mask, score in zip(masks, scores):
            union |= mask
            inter &= mask
            weighted += mask * float(score)
        if len(scores):
            weighted /= sum(map(float, scores))
        return union, inter, weighted

    summaries = []
    for item in models:
        name = item["model"]
        save(out / "status.json", dict(phase="validation", model=name, completed=len(summaries), total=len(models)))
        weights = weights_dir / item["asset"]
        if weights.stat().st_size != int(item["bytes"]) or sha(weights) != item["sha256"]:
            raise ValueError(f"Changed published weights: {name}")
        infer = load_predictor(item, weights)
        threshold = THRESHOLDS[item["architecture"]]
        instance = item["architecture"] != "u-net"
        pixel_rows, detections, grid = [], [], []
        for i, row in enumerate(valid, 1):
            reference, masks, scores = sample(row, infer, 0.001 if instance else threshold)
            union = np.zeros_like(reference)
            for mask, score in zip(masks, scores):
                if score >= threshold:
                    union |= mask
                if instance:
                    encoding = rle(mask)
                    encoding["counts"] = encoding["counts"].decode("ascii")
                    detections.append(dict(image_id=i, category_id=1, segmentation=encoding, score=float(score)))
            if name == FPN:
                retained = [(m, s) for m, s in zip(masks, scores) if s >= threshold]
                _, _, weighted = fuse(reference, [p[0] for p in retained], [p[1] for p in retained])
                bins = np.searchsorted(GRID, weighted, side="right")
                pred = np.bincount(bins.ravel(), minlength=20)[::-1].cumsum()[::-1][1:]
                tp = np.bincount(bins[reference], minlength=20)[::-1].cumsum()[::-1][1:]
                denom = pred + reference.sum() - tp
                grid.append(np.divide(tp, denom, out=np.ones(len(GRID)), where=denom != 0))
            pixel_rows.append(dict(model=name, image=row["image"], **metrics(union, reference)))
            if i % 150 == 0:
                print(f"{name}: validation {i}/892", flush=True)
        summary = summarize(pixel_rows, name, "validation892")
        summary.update(weights_sha256=item["sha256"], selection_images=833, AP=None, AP50=None)
        if instance:
            if detections:
                result = gt.loadRes(detections)
            else:
                result = COCO()
                result.dataset = dict(images=gt.dataset["images"], categories=gt.dataset["categories"], annotations=[])
                result.createIndex()
            scorer = COCOeval(gt, result, "segm")
            scorer.params.maxDets = [1, 10, 100]
            scorer.evaluate()
            scorer.accumulate()
            scorer.summarize()
            summary.update(AP=float(scorer.stats[0]), AP50=float(scorer.stats[1]))
        csv_write(per_image / f"{name}_validation_per_image.csv", pixel_rows)
        summaries.append(summary)
        csv_write(out / "validation_summary.csv", summaries)

        if name == FPN:
            means = np.asarray(grid).mean(axis=0)
            idx = int(np.argmax(means))
            selected = GRID[idx]
            save(out / "fusion_selection.json", dict(threshold=selected, selection_images=892,
                 weights_sha256=item["sha256"], curve=[dict(threshold=t, iou=float(m)) for t, m in zip(GRID, means)]))
            result_rows = []
            for i, row in enumerate(test, 1):
                reference, masks, scores = sample(row, infer, 0.30)
                union, inter, weighted = fuse(reference, masks, scores)
                for method, mask in (("OR", union), ("AND", inter), ("weighted_mean", weighted >= selected)):
                    result_rows.append(dict(image=row["image"], fusion=method, **metrics(mask, reference)))
                if i % 150 == 0:
                    print(f"Fusion: test {i}/892", flush=True)
            baseline = {r["image"]: r for r in csv_read(REPO / "models/mask-rcnn" / FPN / "internal/test_per_image.csv")}
            for r in result_rows:
                if r["fusion"] == "OR" and any(abs(float(baseline[r["image"]][k]) - r[k]) > 1e-8 for k in KEYS):
                    raise ValueError("Fixed-checkpoint OR did not reproduce the released test ledger")
            csv_write(out / "fusion_per_image.csv", result_rows)
            results = [summarize([r for r in result_rows if r["fusion"] == method], method, "test")
                       for method in ("OR", "AND", "weighted_mean")]
            csv_write(out / "fusion_summary.csv", results)
        del infer
        if not instance:
            import keras
            keras.backend.clear_session()
        gc.collect()
        torch.cuda.empty_cache()
    save(out / "status.json", dict(phase="complete", models=len(summaries), training=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--stage", choices=("aggregate", "comparison", "validation"), required=True)
    args = parser.parse_args()
    if args.stage == "validation" and (args.data is None or args.weights is None):
        parser.error("--data and --weights are required for validation")
    out = args.output.resolve()
    protocol = read(MANIFESTS / "partition_protocol.json")
    manifest = MANIFESTS / "partition_manifest.csv"
    if sha(manifest) != protocol["manifest_sha256"] or protocol["baseline_manifest_sha256"] != BASELINE:
        raise ValueError("Unexpected validation892 protocol")
    rows = csv_read(manifest)
    old_manifest = REPO / "data/partition_manifest.csv"
    if sha(old_manifest) != BASELINE:
        raise ValueError("Unexpected published partition")
    old = csv_read(old_manifest)
    if [r for r in rows if r["split"] in ("train", "test")] != [r for r in old if r["split"] in ("train", "test")]:
        raise ValueError("The original train or test split changed")
    if sum(r["split"] == "valid" for r in rows) != 892:
        raise ValueError("Wrong validation population")
    models = csv_read(REPO / "models/checkpoints.csv")
    if len(models) != 17 or any(r["selection_images"] != "833" for r in models):
        raise ValueError("Expected all 17 published validation833 checkpoints")
    out.mkdir(parents=True, exist_ok=False)
    if args.stage == "aggregate":
        aggregate(out, models, old, protocol)
    elif args.stage == "comparison":
        compare_validation(out, models, old, rows)
    else:
        if (out / "status.json").exists():
            raise ValueError("A validation job already owns this output")
        save(out / "status.json", dict(phase="preflight", model=None, completed=0, total=17))
        try:
            evaluate(args.data.resolve(), args.weights.resolve(), out, models, rows)
        except BaseException as error:
            previous = read(out / "status.json")
            save(out / "status.json", dict(previous, phase="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
