"""Build a fixed-size test set by replacing visually overlapping images."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def rasterize_yolo(label_path: Path, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        coordinates = np.asarray(values[1:], dtype=np.float32)
        if coordinates.size % 2:
            continue
        points = coordinates.reshape(-1, 2)
        points[:, 0] = np.clip(np.rint(points[:, 0] * (size - 1)), 0, size - 1)
        points[:, 1] = np.clip(np.rint(points[:, 1] * (size - 1)), 0, size - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask


def mask_to_yolo(mask: np.ndarray) -> list[str]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        epsilon = max(0.5, 0.001 * cv2.arcLength(contour, closed=True))
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        coordinates = []
        for x, y in polygon:
            coordinates.extend((x / (width - 1), y / (height - 1)))
        lines.append("0 " + " ".join(f"{value:.8f}" for value in coordinates))
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_out = args.out_root / "images"
    label_out = args.out_root / "labels"
    mask_out = args.out_root / "ground_truth_masks"
    for directory in (image_out, label_out, mask_out):
        if directory.exists():
            shutil.rmtree(directory)
    for directory in (image_out, label_out, mask_out):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    removed = {Path(row["removed_test_image"]).resolve() for row in manifest}
    output_rows = []

    source_paths = sorted(
        path for path in args.source_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in source_paths:
        if image_path.resolve() in removed:
            continue
        label_path = args.source_labels / f"{image_path.stem}.txt"
        destination_image = image_out / image_path.name
        destination_label = label_out / f"{image_path.stem}.txt"
        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)
        output_rows.append(
            {
                "output_image": str(destination_image),
                "output_label": str(destination_label),
                "origin": "retained_original_test",
                "source_image": str(image_path),
                "source_mask_or_label": str(label_path),
            }
        )

    conversion_ious = []
    for index, row in enumerate(manifest, 1):
        source_image = Path(row["replacement_image"])
        source_mask = Path(row["replacement_mask"])
        stem = f"replacement_{index:03d}_{source_image.stem}"
        destination_image = image_out / f"{stem}.jpg"
        destination_label = label_out / f"{stem}.txt"
        destination_mask = mask_out / f"{stem}.png"

        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read replacement pair: {source_image}, {source_mask}")
        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        binary = (mask >= 128).astype(np.uint8)
        lines = mask_to_yolo(binary)
        cv2.imwrite(str(destination_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(destination_mask), binary * 255)
        destination_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        reconstructed = rasterize_yolo(destination_label, args.size)
        intersection = np.logical_and(binary, reconstructed).sum()
        union = np.logical_or(binary, reconstructed).sum()
        conversion_iou = float(intersection / union) if union else 1.0
        conversion_ious.append(conversion_iou)
        output_rows.append(
            {
                "output_image": str(destination_image),
                "output_label": str(destination_label),
                "origin": "similarity_safe_replacement",
                "source_image": str(source_image),
                "source_mask_or_label": str(source_mask),
            }
        )

    images = sorted(path for path in image_out.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(label_out.glob("*.txt"))
    masks = sorted(mask_out.glob("*.png"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if len(images) != 892 or len(labels) != 892 or len(masks) != len(manifest):
        raise RuntimeError(
            f"Expected 892 image-label pairs and {len(manifest)} replacement masks, "
            f"got {len(images)}, {len(labels)}, {len(masks)}"
        )
    if image_stems != label_stems:
        raise RuntimeError("Image and label stems do not match")

    with (args.out_root / "test_set_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    yaml_path = args.out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {args.out_root.as_posix()}\ntrain: images\nval: images\ntest: images\nnames:\n  0: Floresta\n",
        encoding="utf-8",
    )
    summary = {
        "original_test_images": len(source_paths),
        "removed_unique_test_images": len(removed),
        "retained_original_test_images": len(source_paths) - len(removed),
        "replacement_images": len(manifest),
        "replacement_source_masks": len(masks),
        "final_test_images": len(images),
        "mask_to_yolo_iou_mean": float(np.mean(conversion_ious)),
        "mask_to_yolo_iou_min": float(np.min(conversion_ious)),
    }
    (args.out_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
