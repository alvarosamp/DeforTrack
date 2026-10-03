"""Select distribution-matched replacement images with cross-split cosine below a threshold."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


def source_id(path: str | Path) -> str:
    name = Path(path).stem.lower()
    name = re.sub(r"\.rf\.[0-9a-f]+$", "", name)
    name = re.sub(r"_(jpg|jpeg|png)$", "", name)
    return name


def yolo_coverage(label_path: Path, size: int = 640) -> float:
    mask = np.zeros((size, size), dtype=np.uint8)
    if label_path.exists():
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
    return float(mask.mean())


def mask_coverage(mask_path: Path) -> float:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read {mask_path}")
    return float((mask >= 128).mean())


def polygon_roundtrip_iou(mask_path: Path) -> float:
    grayscale = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if grayscale is None:
        raise RuntimeError(f"Could not read {mask_path}")
    binary = (grayscale >= 128).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reconstructed = np.zeros_like(binary)
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        epsilon = max(0.5, 0.001 * cv2.arcLength(contour, closed=True))
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
        cv2.fillPoly(reconstructed, [polygon], 1)
    intersection = np.logical_and(binary, reconstructed).sum()
    union = np.logical_or(binary, reconstructed).sum()
    return float(intersection / union) if union else 1.0


class CandidateDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = ResNet18_Weights.DEFAULT.transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path)


def extract(paths: list[Path], cache: Path, batch_size: int) -> torch.Tensor:
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved["paths"] == [str(path) for path in paths]:
            return saved["features"].float()

    loader = DataLoader(CandidateDataset(paths), batch_size=batch_size, shuffle=False, num_workers=0)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    features = []
    with torch.inference_mode():
        for index, (images, _) in enumerate(loader, 1):
            embedded = torch.nn.functional.normalize(model(images), dim=1)
            features.append(embedded.cpu())
            if index % 10 == 0 or index == len(loader):
                print(f"candidate embeddings: {index}/{len(loader)}", flush=True)
    result = torch.cat(features)
    torch.save(
        {"features": result.half(), "paths": [str(path) for path in paths]}, cache
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-cache", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--flagged-test", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split = torch.load(args.split_cache, map_location="cpu", weights_only=False)
    split_features = split["features"].float()
    split_roles = np.asarray(split["roles"])
    split_paths = np.asarray(split["paths"])
    existing_ids = {source_id(path) for path in split_paths}

    metadata = pd.read_csv(args.candidate_root / "meta_data.csv")
    candidates = []
    masks = []
    for row in metadata.itertuples(index=False):
        image = args.candidate_root / "images" / row.image
        mask = args.candidate_root / "masks" / row.mask
        if image.exists() and mask.exists():
            candidates.append(image)
            masks.append(mask)

    features = extract(candidates, args.out_dir / "candidate_embeddings.pt", args.batch_size)
    maximum = torch.full((len(candidates),), -1.0)
    nearest_index = torch.zeros(len(candidates), dtype=torch.long)
    for start in range(0, len(features), 128):
        similarities = features[start : start + 128] @ split_features.T
        values, indices = similarities.max(dim=1)
        maximum[start : start + len(values)] = values
        nearest_index[start : start + len(values)] = indices

    candidate_rows = []
    eligible_indices = []
    for index, (image, mask) in enumerate(zip(candidates, masks)):
        nearest = int(nearest_index[index])
        same_source = source_id(image) in existing_ids
        roundtrip_iou = polygon_roundtrip_iou(mask)
        row = {
            "candidate_image": str(image),
            "candidate_mask": str(mask),
            "source_id": source_id(image),
            "same_source_name_in_existing_splits": same_source,
            "nearest_existing_role": split_roles[nearest],
            "nearest_existing_path": split_paths[nearest],
            "max_cosine_to_existing_splits": float(maximum[index]),
            "forest_coverage": mask_coverage(mask),
            "mask_to_polygon_iou": roundtrip_iou,
            "eligible": (
                not same_source
                and float(maximum[index]) < args.threshold
                and roundtrip_iou >= 0.98
            ),
        }
        candidate_rows.append(row)
        if row["eligible"]:
            eligible_indices.append(index)

    pd.DataFrame(candidate_rows).to_csv(
        args.out_dir / "candidate_similarity_audit.csv", index=False
    )

    flagged_paths = [
        Path(line.strip())
        for line in args.flagged_test.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    removed = [
        {
            "removed_test_image": str(path),
            "removed_test_label": str(args.test_labels / f"{path.stem}.txt"),
            "removed_forest_coverage": yolo_coverage(args.test_labels / f"{path.stem}.txt"),
        }
        for path in flagged_paths
    ]
    removed.sort(key=lambda row: row["removed_forest_coverage"])

    selected: list[int] = []
    manifest = []
    for removed_row in removed:
        target = removed_row["removed_forest_coverage"]
        options = sorted(
            (index for index in eligible_indices if index not in selected),
            key=lambda index: abs(candidate_rows[index]["forest_coverage"] - target),
        )
        chosen = None
        for index in options:
            if not selected:
                chosen = index
                break
            pairwise = features[index] @ features[selected].T
            if float(pairwise.max()) < args.threshold:
                chosen = index
                break
        if chosen is None:
            raise RuntimeError("No similarity-safe replacement remained")
        selected.append(chosen)
        candidate = candidate_rows[chosen]
        manifest.append(
            {
                **removed_row,
                "replacement_image": candidate["candidate_image"],
                "replacement_mask": candidate["candidate_mask"],
                "replacement_forest_coverage": candidate["forest_coverage"],
                "coverage_delta": candidate["forest_coverage"] - target,
                "replacement_max_cosine_to_existing_splits": candidate[
                    "max_cosine_to_existing_splits"
                ],
                "replacement_nearest_existing_role": candidate["nearest_existing_role"],
                "replacement_nearest_existing_path": candidate["nearest_existing_path"],
            }
        )

    with (args.out_dir / "replacements_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    print(
        f"candidates={len(candidates)} eligible={len(eligible_indices)} "
        f"selected={len(selected)} max_selected_cosine={max(maximum[selected]).item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
