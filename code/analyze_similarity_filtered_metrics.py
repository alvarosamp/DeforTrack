"""Recompute per-image segmentation metrics after a similarity-based exclusion.

The analysis is a sensitivity check, not a replacement for a source-aware test split.
It removes final-test queries whose nearest training neighbor reaches the configured
cosine-similarity threshold and reports the change from the complete test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath

import pandas as pd


METRICS = [
    "iou",
    "dice_f1",
    "precision",
    "recall",
    "boundary_iou",
    "boundary_f1",
    "pixel_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threshold", type=float, default=0.98)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    similarity_path = args.root / "results" / "split_similarity" / "nearest_cross_split.csv"
    internal_dir = args.root / "results" / "internal_test_892"
    output_dir = args.root / "results" / "similarity_filtered_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    similarity = pd.read_csv(similarity_path)
    test_to_train = similarity[
        (similarity["query_split"] == "test")
        & (similarity["reference_split"] == "train")
    ].copy()
    test_to_train["image"] = test_to_train["query"].map(
        lambda value: PureWindowsPath(value).name
    )
    excluded = test_to_train[test_to_train["cosine"] >= args.threshold].copy()
    excluded_names = set(excluded["image"])

    input_files = [
        internal_dir / "yolo_test_per_image.csv",
        internal_dir / "maskrcnn_test_per_image.csv",
        internal_dir / "unet_892_per_image.csv",
    ]
    frames = [pd.read_csv(path) for path in input_files]
    per_image = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, object]] = []
    for model, model_rows in per_image.groupby("model", sort=False):
        retained = model_rows[~model_rows["image"].isin(excluded_names)]
        row: dict[str, object] = {
            "model": model,
            "similarity_threshold": args.threshold,
            "n_complete": len(model_rows),
            "n_excluded": len(model_rows) - len(retained),
            "n_retained": len(retained),
        }
        for metric in METRICS:
            complete_mean = model_rows[metric].mean() * 100
            retained_mean = retained[metric].mean() * 100
            row[f"{metric}_complete_pct"] = complete_mean
            row[f"{metric}_filtered_pct"] = retained_mean
            row[f"{metric}_delta_pp"] = retained_mean - complete_mean
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "similarity_filtered_metrics.csv", index=False)
    excluded[["image", "query", "reference", "cosine"]].sort_values(
        "cosine", ascending=False
    ).to_csv(output_dir / "excluded_test_images.csv", index=False)

    print(
        f"Excluded {len(excluded_names)} of {len(test_to_train)} test images at "
        f"cosine >= {args.threshold:.2f}; retained {len(test_to_train) - len(excluded_names)}."
    )
    print(summary[["model", "iou_complete_pct", "iou_filtered_pct", "iou_delta_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
