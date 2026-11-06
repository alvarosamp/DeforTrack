"""Verify the recomposed test set against cached train/validation embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


class Images(Dataset):
    def __init__(self, root: Path) -> None:
        self.paths = sorted(
            path
            for path in root.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.transform = ResNet18_Weights.DEFAULT.transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--split-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def extract(images: Path, batch_size: int) -> tuple[torch.Tensor, list[str]]:
    dataset = Images(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    features: list[torch.Tensor] = []
    paths: list[str] = []
    with torch.inference_mode():
        for index, (batch, batch_paths) in enumerate(loader, 1):
            features.append(torch.nn.functional.normalize(model(batch), dim=1))
            paths.extend(batch_paths)
            print(f"embedding batch {index}/{len(loader)}", flush=True)
    return torch.cat(features), paths


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_features, final_paths = extract(args.images, args.batch_size)
    cached = torch.load(args.split_cache, map_location="cpu", weights_only=False)
    source_features = cached["features"].float()
    source_roles = np.asarray(cached["roles"])
    source_paths = np.asarray(cached["paths"])

    nearest_rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    flagged_test: set[str] = set()
    for role in ("train", "validation"):
        indices = np.where(source_roles == role)[0]
        reference = source_features[indices]
        pair_count = 0
        scores: list[float] = []
        for start in range(0, len(final_features), 128):
            similarities = final_features[start : start + 128] @ reference.T
            values, locations = similarities.max(dim=1)
            pair_count += int((similarities >= args.threshold).sum())
            for offset, (value, location) in enumerate(zip(values, locations)):
                score = float(value)
                test_path = final_paths[start + offset]
                scores.append(score)
                if score >= args.threshold:
                    flagged_test.add(test_path)
                nearest_rows.append(
                    {
                        "comparison": f"final-test-{role}",
                        "test_image": test_path,
                        "reference_image": source_paths[indices[int(location)]],
                        "cosine": score,
                        "at_or_above_threshold": score >= args.threshold,
                    }
                )
        values = np.asarray(scores)
        comparisons.append(
            {
                "comparison": f"final-test-{role}",
                "test_images": len(final_paths),
                "reference_images": len(indices),
                "threshold": args.threshold,
                "pairs_at_or_above_threshold": pair_count,
                "test_images_with_nearest_at_or_above_threshold": int(
                    (values >= args.threshold).sum()
                ),
                "nearest_mean": float(values.mean()),
                "nearest_p95": float(np.quantile(values, 0.95)),
                "nearest_p99": float(np.quantile(values, 0.99)),
                "nearest_max": float(values.max()),
            }
        )

    reference_hashes: set[str] = set()
    for role, path in zip(source_roles, source_paths):
        if role in {"train", "validation"}:
            reference_hashes.add(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    exact_duplicates = sum(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() in reference_hashes
        for path in final_paths
    )

    with (args.out_dir / "final_test_nearest_similarity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(nearest_rows[0]))
        writer.writeheader()
        writer.writerows(nearest_rows)
    summary = {
        "extractor": "torchvision ResNet-18 ImageNet-1K V1, 512-D penultimate features",
        "metric": "cosine similarity",
        "final_test_images": len(final_paths),
        "unique_test_images_at_or_above_threshold": len(flagged_test),
        "exact_binary_duplicates_to_train_or_validation": exact_duplicates,
        "comparisons": comparisons,
    }
    (args.out_dir / "final_similarity_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
