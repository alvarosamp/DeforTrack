"""Detectron2 records for the single-class forest masks."""
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from common.validation import verify_live_file


def instance_masks(text, height, width):
    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions")
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 7 or len(fields[1:]) % 2 or fields[0] != "0":
            raise ValueError(f"Invalid single-class polygon at line {number}")
        points = np.asarray(fields[1:], dtype=np.float32).reshape(-1, 2)
        # Roboflow exports contain six polygons with negative roundoff around 1e-17.
        if not np.isfinite(points).all() or (points < -1e-12).any() or (points > 1 + 1e-12).any():
            raise ValueError(f"Invalid normalized coordinates at line {number}")
        points = np.clip(points, 0, 1)
        points[:, 0] = np.rint(points[:, 0] * (width - 1))
        points[:, 1] = np.rint(points[:, 1] * (height - 1))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        if not mask.any():
            raise ValueError(f"Empty rasterized polygon at line {number}")
        result.append(mask)
    return result


def instance_annotation(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Empty instance")
    # Detectron2 uses zero-based categories and XYWH_ABS (BoxMode value 1).
    return dict(category_id=0, bbox_mode=1,
                bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                      int(ys.max() - ys.min() + 1)],
                segmentation=mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))),
                iscrowd=0)


def build_records(rows, role, data_root, expected_count):
    if role not in {"train", "valid"}:
        raise ValueError("Training adapter must not register the test partition")
    selected = [row for row in rows if row["split"] == role]
    if len(selected) != expected_count:
        raise ValueError("Incomplete partition")
    if len({row["image"] for row in selected}) != len(selected):
        raise ValueError("Duplicate image in partition")
    records = []
    for index, row in enumerate(selected):
        for key in ("image", "label"):
            if Path(row[key]).name != row[key] or "/" in row[key] or "\\" in row[key]:
                raise ValueError("Escaping manifest filename")
        image = Path(data_root) / role / "images" / row["image"]
        label = Path(data_root) / role / "labels" / row["label"]
        verify_live_file(image, row["image_sha256"])
        verify_live_file(label, row["label_sha256"])
        decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Unreadable image: {image}")
        height, width = decoded.shape[:2]
        if (height, width) != (640, 640):
            raise ValueError(f"Unexpected exported resolution: {image}")
        try:
            masks = instance_masks(label.read_text(encoding="utf-8"), height, width)
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error
        records.append(dict(file_name=str(image), image_id=index + 1,
                            height=height, width=width,
                            annotations=[instance_annotation(mask) for mask in masks]))
        if (index + 1) % 1000 == 0:
            print(f"Mask R-CNN adapter {role}: {index + 1}/{expected_count}", flush=True)
    return records
