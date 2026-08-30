"""Consolidate the standardized 892-image metrics used by the manuscript."""

import argparse
import csv
import json
import math
from pathlib import Path


DISPLAY_NAMES = {
    "mask_rcnn_R_101_C4_3x": "ResNet-101 C4 3x",
    "mask_rcnn_R_101_DC5_3x": "ResNet-101 DC5 3x",
    "mask_rcnn_R_101_FPN_3x": "ResNet-101 FPN 3x",
    "mask_rcnn_R_50_C4_1x": "ResNet-50 C4 1x",
    "mask_rcnn_R_50_C4_3x": "ResNet-50 C4 3x",
    "mask_rcnn_R_50_DC5_1x": "ResNet-50 DC5 1x",
}

ORDER = [
    "YOLOv8x", "YOLOv8l", "YOLOv8m", "YOLOv8s", "YOLOv8n",
    "YOLOv11x", "YOLOv11l", "YOLOv11m", "YOLOv11s", "YOLOv11n",
    "mask_rcnn_R_101_FPN_3x", "mask_rcnn_R_101_C4_3x",
    "mask_rcnn_R_101_DC5_3x", "mask_rcnn_R_50_C4_3x",
    "mask_rcnn_R_50_C4_1x", "mask_rcnn_R_50_DC5_1x", "U-Net",
]


def read_csv(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_test_count(row: dict) -> None:
    count = int(row.get("n", row.get("images", 0)))
    if count != 892:
        raise ValueError(f"{row.get('model')}: expected 892 images, found {count}")


def percent(row: dict, key: str) -> float:
    return round(float(row[key]) * 100.0, 2)


def uncertainty(row: dict, metric: str) -> tuple[float, float]:
    std = float(row[f"{metric}_std"])
    ci_key = f"{metric}_ci95_halfwidth"
    ci = float(row[ci_key]) if ci_key in row and row[ci_key] else 1.96 * std / math.sqrt(892)
    return round(std * 100.0, 2), round(ci * 100.0, 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-summary", required=True)
    parser.add_argument("--mask-summary", required=True)
    parser.add_argument("--mask-map", required=True)
    parser.add_argument("--unet-summary", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    yolo = {r["model"]: r for r in read_csv(args.yolo_summary)}
    mask = {r["model"]: r for r in read_csv(args.mask_summary)}
    unet_rows = read_csv(args.unet_summary)
    if len(unet_rows) != 1:
        raise ValueError("Expected exactly one U-Net summary row")
    unet = {unet_rows[0]["model"]: unet_rows[0]}
    map_rows = json.loads(Path(args.mask_map).read_text(encoding="utf-8"))
    mask_map = {r["model"]: r for r in map_rows}

    all_rows = yolo | mask | unet
    if set(all_rows) != set(ORDER):
        missing = set(ORDER) - set(all_rows)
        extra = set(all_rows) - set(ORDER)
        raise ValueError(f"Model set mismatch; missing={missing}, extra={extra}")

    segmentation = []
    for name in ORDER:
        row = all_rows[name]
        require_test_count(row)
        iou_std, iou_ci = uncertainty(row, "iou")
        dice_std, dice_ci = uncertainty(row, "dice_f1")
        if name.startswith("YOLO"):
            map50 = round(float(row["map_50"]) * 100.0, 2)
            map5095 = round(float(row["map_50_95"]) * 100.0, 2)
            family = "YOLOv8" if name.startswith("YOLOv8") else "YOLOv11"
        elif name == "U-Net":
            map50 = "NA"
            map5095 = "NA"
            family = "U-Net"
        else:
            map50 = round(float(mask_map[name]["map_50"]) * 100.0, 2)
            map5095 = round(float(mask_map[name]["map_50_95"]) * 100.0, 2)
            family = "Mask R-CNN"
        segmentation.append({
            "family": family,
            "model": DISPLAY_NAMES.get(name, name),
            "images": 892,
            "iou_percent": percent(row, "iou_mean"),
            "iou_std_percent": iou_std,
            "iou_ci95_halfwidth_percent": iou_ci,
            "map_50_percent": map50,
            "map_50_95_percent": map5095,
            "dice_f1_percent": percent(row, "dice_f1_mean"),
            "dice_f1_std_percent": dice_std,
            "dice_f1_ci95_halfwidth_percent": dice_ci,
            "boundary_iou_percent": percent(row, "boundary_iou_mean"),
            "boundary_f1_percent": percent(row, "boundary_f1_mean"),
            "pixel_accuracy_percent": percent(row, "pixel_accuracy_mean"),
            "precision_percent": percent(row, "precision_mean"),
            "recall_percent": percent(row, "recall_mean"),
        })

    profile_rows = read_csv(args.profile)
    profiles = {r["model"]: r for r in profile_rows}
    if set(profiles) != set(ORDER):
        missing = set(ORDER) - set(profiles)
        extra = set(profiles) - set(ORDER)
        raise ValueError(f"Profile set mismatch; missing={missing}, extra={extra}")
    computational = []
    for name in ORDER:
        row = profiles[name]
        require_test_count(row)
        if int(row.get("warmup_images", 0)) != 892:
            raise ValueError(f"{name}: expected one 892-image warm-up pass")
        if int(row.get("repetitions", 0)) != 3:
            raise ValueError(f"{name}: expected 3 profile repetitions")
        if int(row.get("measured_inferences", 0)) != 2676:
            raise ValueError(f"{name}: expected 2676 measured inferences")
        computational.append({
            "family": ("YOLOv8" if name.startswith("YOLOv8") else
                       "YOLOv11" if name.startswith("YOLOv11") else
                       "U-Net" if name == "U-Net" else "Mask R-CNN"),
            "model": DISPLAY_NAMES.get(name, name),
            "images": 892,
            "model_size_mb": round(float(row["model_size_mb"]), 2),
            "peak_process_ram_mb": round(float(row["ram_peak_mb"]), 2),
            "latency_s_per_image": round(float(row["inference_time_s_per_image"]), 4),
            "trial_latency_s_std": round(float(row["trial_latency_s_std"]), 4),
            "fps": round(float(row["fps"]), 2),
            "normalized_cpu_percent": round(float(row["cpu_mean_percent_total"]), 2),
            "gpu_energy_j_per_image": round(float(row["energy_j_per_image"]), 3),
        })

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "segmentation_metrics_892.csv", segmentation)
    write_csv(out_dir / "computational_metrics_892.csv", computational)
    print(f"Validated and wrote {len(segmentation)} segmentation and {len(computational)} profile rows")


if __name__ == "__main__":
    main()
