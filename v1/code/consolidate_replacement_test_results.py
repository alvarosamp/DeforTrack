"""Consolidate the 17-model evaluation on the similarity-safe 892-image test."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ORDER = [
    "YOLOv8x", "YOLOv8l", "YOLOv8m", "YOLOv8s", "YOLOv8n",
    "YOLOv11x", "YOLOv11l", "YOLOv11m", "YOLOv11s", "YOLOv11n",
    "mask_rcnn_R_101_FPN_3x", "mask_rcnn_R_101_C4_3x",
    "mask_rcnn_R_101_DC5_3x", "mask_rcnn_R_50_C4_3x",
    "mask_rcnn_R_50_C4_1x", "mask_rcnn_R_50_DC5_1x", "U-Net",
]

DISPLAY = {
    "mask_rcnn_R_101_FPN_3x": "ResNet-101 FPN 3x",
    "mask_rcnn_R_101_C4_3x": "ResNet-101 C4 3x",
    "mask_rcnn_R_101_DC5_3x": "ResNet-101 DC5 3x",
    "mask_rcnn_R_50_C4_3x": "ResNet-50 C4 3x",
    "mask_rcnn_R_50_C4_1x": "ResNet-50 C4 1x",
    "mask_rcnn_R_50_DC5_1x": "ResNet-50 DC5 1x",
}

METRICS = [
    "iou", "dice_f1", "boundary_iou", "boundary_f1",
    "pixel_accuracy", "precision", "recall", "inference_time_s",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--maskrcnn", type=Path, required=True)
    parser.add_argument("--unet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--computational-output", type=Path)
    args = parser.parse_args()

    rows = read(args.yolo) + read(args.maskrcnn) + read(args.unet)
    by_model = {row["model"]: row for row in rows}
    if set(by_model) != set(ORDER):
        raise RuntimeError(
            f"Model mismatch: missing={set(ORDER) - set(by_model)}, "
            f"extra={set(by_model) - set(ORDER)}"
        )

    output = []
    for model in ORDER:
        row = by_model[model]
        n = int(row["n"])
        if n != 892:
            raise RuntimeError(f"{model}: expected 892 images, got {n}")
        result: dict[str, object] = {
            "model": DISPLAY.get(model, model),
            "checkpoint_key": model,
            "images": n,
            "map_50_percent": (
                round(100 * float(row["map_50"]), 2) if row.get("map_50") else "NA"
            ),
            "map_50_95_percent": (
                round(100 * float(row["map_50_95"]), 2)
                if row.get("map_50_95") else "NA"
            ),
        }
        for metric in METRICS:
            mean = float(row[f"{metric}_mean"])
            standard_deviation = float(row[f"{metric}_std"])
            ci = (
                float(row[f"{metric}_ci95_halfwidth"])
                if row.get(f"{metric}_ci95_halfwidth")
                else 1.96 * standard_deviation / math.sqrt(n)
            )
            scale = 1 if metric == "inference_time_s" else 100
            result[f"{metric}_mean"] = round(scale * mean, 6)
            result[f"{metric}_std"] = round(scale * standard_deviation, 6)
            result[f"{metric}_ci95_halfwidth"] = round(scale * ci, 6)
        result.update(
            {
                "iou_percent": round(float(result["iou_mean"]), 2),
                "dice_f1_percent": round(float(result["dice_f1_mean"]), 2),
                "boundary_iou_percent": round(float(result["boundary_iou_mean"]), 2),
                "boundary_f1_percent": round(float(result["boundary_f1_mean"]), 2),
                "pixel_accuracy_percent": round(float(result["pixel_accuracy_mean"]), 2),
                "precision_percent": round(float(result["precision_mean"]), 2),
                "recall_percent": round(float(result["recall_mean"]), 2),
            }
        )
        output.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} models to {args.output}")

    if bool(args.profile) != bool(args.computational_output):
        raise RuntimeError("--profile and --computational-output must be used together")
    if args.profile:
        profile_rows = {row["model"]: row for row in read(args.profile)}
        if set(profile_rows) != set(ORDER):
            raise RuntimeError("Computational profile does not contain all 17 checkpoints")
        computational = []
        for model in ORDER:
            row = profile_rows[model]
            if (
                int(row["images"]) != 892
                or int(row["warmup_images"]) != 892
                or int(row["repetitions"]) != 3
                or int(row["measured_inferences"]) != 2676
            ):
                raise RuntimeError(f"Invalid profiling protocol for {model}")
            computational.append(
                {
                    "model": DISPLAY.get(model, model),
                    "checkpoint_key": model,
                    "images": 892,
                    "model_size_mb": round(float(row["model_size_mb"]), 2),
                    "peak_process_ram_mb": round(float(row["ram_peak_mb"]), 2),
                    "latency_s_per_image": round(
                        float(row["inference_time_s_per_image"]), 4
                    ),
                    "trial_latency_s_std": round(float(row["trial_latency_s_std"]), 4),
                    "fps": round(float(row["fps"]), 2),
                    "normalized_cpu_percent": round(
                        float(row["cpu_mean_percent_total"]), 2
                    ),
                    "gpu_energy_j_per_image": round(
                        float(row["energy_j_per_image"]), 3
                    ),
                }
            )
        args.computational_output.parent.mkdir(parents=True, exist_ok=True)
        with args.computational_output.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(computational[0]))
            writer.writeheader()
            writer.writerows(computational)
        print(
            f"wrote {len(computational)} models to {args.computational_output}"
        )


if __name__ == "__main__":
    main()
