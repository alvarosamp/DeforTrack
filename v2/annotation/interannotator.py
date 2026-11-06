"""Agreement between the three annotation exports (IoU, Dice, precision, Kappa)."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np


SOURCES = {"A1": None, "A2": None, "A3": None}
OUT = None
SIZE = 640
MAX_IMAGE_MSE = 0.002


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_key(name):
    stem = re.sub(r"\.rf\.[^.]+$", "", PurePosixPath(name).stem.lower())
    return re.sub(r"(?:(?:_jpg)|(?:_jpeg)|(?:_png))+$", "", stem)


def thumbnail(data):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode source image")
    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def inventory(archive, annotator):
    records = defaultdict(list)
    entries = set(archive.namelist())
    if annotator == "A1":
        for name in sorted(entries):
            if "/images/" not in name or PurePosixPath(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if label not in entries:
                raise ValueError(f"Missing annotation: {label}")
            image_bytes = archive.read(name)
            label_bytes = archive.read(label)
            polygons = []
            for line in label_bytes.decode().splitlines():
                values = line.split()
                if not values:
                    continue
                if values[0] != "0" or len(values) < 7 or (len(values) - 1) % 2:
                    raise ValueError(f"Invalid forest annotation: {label}")
                polygons.append(np.asarray(values[1:], dtype=np.float32).reshape(-1, 2))
            records[source_key(name)].append({"entry": name, "annotation_entry": label, "annotation_sha256": digest(label_bytes), "image_sha256": digest(image_bytes), "thumbnail": thumbnail(image_bytes), "polygons": polygons})
    else:
        for annotation_entry in sorted(name for name in entries if name.endswith("_annotations.coco.json")):
            annotation_bytes = archive.read(annotation_entry)
            data = json.loads(annotation_bytes)
            foreground = {category["id"] for category in data["categories"] if category["name"].lower() in {"forest", "floresta"}}
            if not foreground:
                raise ValueError(f"No forest category: {annotation_entry}")
            annotations = defaultdict(list)
            for annotation in data["annotations"]:
                if annotation["category_id"] not in foreground:
                    raise ValueError(f"Unexpected annotation category: {annotation_entry}")
                annotations[annotation["image_id"]].append(annotation)
            for image in data["images"]:
                entry = str(PurePosixPath(annotation_entry).parent / image["file_name"])
                image_bytes = archive.read(entry)
                polygons = []
                for annotation in annotations[image["id"]]:
                    if not isinstance(annotation["segmentation"], list):
                        raise ValueError(f"Non-polygon COCO segmentation: {entry}")
                    for polygon in annotation["segmentation"]:
                        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                        points[:, 0] /= image["width"]
                        points[:, 1] /= image["height"]
                        polygons.append(points)
                key = source_key(image.get("extra", {}).get("name", image["file_name"]))
                records[key].append({"entry": entry, "annotation_entry": annotation_entry, "annotation_sha256": digest(annotation_bytes), "image_id": image["id"], "image_sha256": digest(image_bytes), "thumbnail": thumbnail(image_bytes), "polygons": polygons})
    print(f"{annotator}: {sum(map(len, records.values()))} items, {len(records)} source names", flush=True)
    return records


def mask(record):
    result = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for polygon in record["polygons"]:
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError(f"Invalid polygon: {record['entry']}")
        points = np.clip(np.rint(polygon * (SIZE - 1)), 0, SIZE - 1).astype(np.int32)
        cv2.fillPoly(result, [points], 1)
    return result.astype(bool)


def scores(reference, prediction):
    tp = np.logical_and(reference, prediction).sum()
    fp = np.logical_and(~reference, prediction).sum()
    fn = np.logical_and(reference, ~prediction).sum()
    tn = np.logical_and(~reference, ~prediction).sum()
    n = reference.size
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    observed = float(tp + tn) / n
    ref_positive = float(tp + fn) / n
    pred_positive = float(tp + fp) / n
    expected = ref_positive * pred_positive + (1 - ref_positive) * (1 - pred_positive)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "iou": float(tp / union) if union else 1.0,
        "dice": float(2 * tp / dice_denominator) if dice_denominator else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else (1.0 if not reference.any() else 0.0),
        "kappa": float((observed - expected) / (1 - expected)) if 1 - expected > 1e-12 else None,
    }


def summarize(rows):
    result = {"n_images": len(rows)}
    for metric in ("iou", "dice", "precision", "kappa"):
        values = np.asarray([row[metric] for row in rows if row[metric] is not None], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite metric: {metric}")
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[metric] = {"n_defined": len(values), "mean": float(values.mean()), "std": std, "ci95_halfwidth": 1.96 * std / math.sqrt(len(values))}
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=Path, required=True, help="YOLO export (annotator 1)")
    parser.add_argument("--a2", type=Path, required=True, help="COCO export (annotator 2)")
    parser.add_argument("--a3", type=Path, required=True, help="COCO export (annotator 3)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    SOURCES.update(A1=args.a1, A2=args.a2, A3=args.a3)
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    inventories, archive_hashes = {}, {}
    for annotator, path in SOURCES.items():
        archive_hashes[annotator] = {"path": path.name, "sha256": digest(path.read_bytes())}
        with ZipFile(path) as archive:
            inventories[annotator] = inventory(archive, annotator)
    common = sorted(set.intersection(*(set(items) for items in inventories.values())))
    manifest, pair_rows, consensus_rows, image_average_rows, matched_polygons = [], [], [], [], []
    for index, key in enumerate(common, 1):
        best = None
        for a in inventories["A1"][key]:
            for b in inventories["A2"][key]:
                for c in inventories["A3"][key]:
                    errors = [float(np.mean((x["thumbnail"] - y["thumbnail"]) ** 2)) for x, y in ((a, b), (a, c), (b, c))]
                    rank = (sum(errors), a["entry"], b["entry"], c["entry"])
                    if best is None or rank < best[0]:
                        best = (rank, [a, b, c], errors)
        _, chosen, errors = best
        entry = {"source_key": key, "maximum_pair_thumbnail_mse": max(errors), "accepted": max(errors) <= MAX_IMAGE_MSE}
        for annotator, record in zip(SOURCES, chosen):
            entry.update({f"{annotator}_{field}": value for field, value in record.items() if field not in {"thumbnail", "polygons"}})
            entry[f"{annotator}_candidate_count"] = len(inventories[annotator][key])
        manifest.append(entry)
        if not entry["accepted"]:
            continue
        masks = [mask(record) for record in chosen]
        matched_polygons.append({"source_key": key, **{annotator: [polygon.tolist() for polygon in record["polygons"]] for annotator, record in zip(SOURCES, chosen)}})
        for annotator, annotation in zip(SOURCES, masks):
            entry[f"{annotator}_mask_sha256"] = digest(np.packbits(annotation).tobytes())
        comparisons = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            row = {"source_key": key, "reference": f"A{a+1}", "prediction": f"A{b+1}", **scores(masks[a], masks[b])}
            pair_rows.append(row)
            comparisons.append(row)
        average = {"source_key": key}
        for metric in ("iou", "dice", "precision", "kappa"):
            values = [row[metric] for row in comparisons if row[metric] is not None]
            average[metric] = sum(values) / len(values) if values else None
        image_average_rows.append(average)
        consensus = np.sum(masks, axis=0) >= 2
        for annotator, annotation in zip(SOURCES, masks):
            consensus_rows.append({"source_key": key, "annotator": annotator, **scores(consensus, annotation)})
        if index % 25 == 0:
            print(f"Matched and scored {index}/{len(common)} source names", flush=True)
    write_csv(OUT / "matching_manifest.csv", manifest)
    if not pair_rows:
        raise ValueError("No geometrically matched image triplets")
    write_csv(OUT / "pair_per_image.csv", pair_rows)
    write_csv(OUT / "image_average_per_image.csv", image_average_rows)
    write_csv(OUT / "consensus_per_image.csv", consensus_rows)
    (OUT / "matched_polygons.json").write_text(json.dumps(matched_polygons), encoding="utf-8")
    report = {
        "archives": archive_hashes, "code_sha256": digest(Path(__file__).read_bytes()),
        "annotators": {"A1": "Alvaro archive", "A2": "Greg archive", "A3": "Luigi archive"},
        "common_source_names": len(common), "accepted_source_names": len(image_average_rows),
        "unmatched_or_missing_source_names": {key: sorted(set(value) - set(common)) for key, value in inventories.items()},
        "geometry_rejected_source_names": [row["source_key"] for row in manifest if not row["accepted"]],
        "protocol": {"canvas": [SIZE, SIZE], "rasterization": "normalized float32 polygons rounded at (size-1), clipped, union", "source_matching": "strip Roboflow token and export extension suffixes; compare all same-name image triplets; minimum total 64x64 RGB MSE, lexical tie-break; no mask metrics used to select", "maximum_pair_image_mse": MAX_IMAGE_MSE, "precision_reference": "lower-index annotator is reference", "empty_region_score": "IoU/Dice=1 for two empty masks; undefined precision=1 only if reference empty, else 0", "kappa_undefined": "expected agreement equal to 1: omit Kappa and report its defined n", "final_table_row": "mean and sample SD across per-image averages of pair scores, not average of pair SDs", "consensus": "two-of-three majority; related to constituent annotators, not independent ground truth"},
        "pairs": {f"A{a}-A{b}": summarize([row for row in pair_rows if row["reference"] == f"A{a}" and row["prediction"] == f"A{b}"]) for a, b in ((1, 2), (1, 3), (2, 3))},
        "per_image_pair_average": summarize(image_average_rows),
        "pooled_pair_image_descriptive": summarize(pair_rows),
        "consensus": {annotator: summarize([row for row in consensus_rows if row["annotator"] == annotator]) for annotator in SOURCES},
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("common_source_names", "accepted_source_names", "geometry_rejected_source_names", "pairs", "per_image_pair_average", "consensus")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
