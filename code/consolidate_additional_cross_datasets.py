import csv
from pathlib import Path


ROOT = Path("outputs/cross_dataset")

DATASETS = {
    "Forest Aerial Images": {
        "note": "forest/non-forest segmentation, external dataset previously evaluated",
        "paths": [
            ("all_images_yolo/cross_dataset_yolo_summary.csv", "YOLO"),
            ("all_images_unet/cross_dataset_unet_summary.csv", "U-Net"),
            ("all_images_detectron/cross_dataset_detectron_summary.csv", "Detectron2"),
        ],
    },
    "Amazon/Atlantic Forest": {
        "note": "forest/non-forest segmentation, closest external validation set",
        "paths": [
            ("amazon_atlantic_yolo/cross_dataset_yolo_summary.csv", "YOLO"),
            ("amazon_atlantic_unet/cross_dataset_unet_summary.csv", "U-Net"),
            ("amazon_atlantic_detectron/cross_dataset_detectron_summary.csv", "Detectron2"),
        ],
    },
    "Dead Tree Kaggle": {
        "note": "standing dead tree segmentation; different target semantics",
        "paths": [
            ("dead_tree_yolo/cross_dataset_yolo_summary.csv", "YOLO"),
            ("dead_tree_unet/cross_dataset_unet_summary.csv", "U-Net"),
            ("dead_tree_detectron/cross_dataset_detectron_summary.csv", "Detectron2"),
        ],
    },
    "Forest-Change": {
        "note": "deforestation change masks from post-change images; different protocol",
        "paths": [
            ("forest_change_yolo/cross_dataset_yolo_summary.csv", "YOLO"),
            ("forest_change_unet/cross_dataset_unet_summary.csv", "U-Net"),
            ("forest_change_detectron/cross_dataset_detectron_summary.csv", "Detectron2"),
        ],
    },
}


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pct(row, key):
    return float(row[key]) * 100.0


def main():
    rows = []
    for dataset, info in DATASETS.items():
        for rel_path, family in info["paths"]:
            for row in read_rows(ROOT / rel_path):
                rows.append(
                    {
                        "dataset": dataset,
                        "model_family": family,
                        "model": row["model"],
                        "n": row["n"],
                        "iou_percent": f"{pct(row, 'iou_mean'):.2f}",
                        "dice_percent": f"{pct(row, 'dice_mean'):.2f}",
                        "precision_percent": f"{pct(row, 'precision_mean'):.2f}",
                        "recall_percent": f"{pct(row, 'recall_mean'):.2f}",
                        "pixel_accuracy_percent": f"{pct(row, 'pixel_accuracy_mean'):.2f}",
                        "note": info["note"],
                    }
                )

    out_csv = ROOT / "additional_cross_dataset_summary.csv"
    out_md = ROOT / "additional_cross_dataset_summary.md"
    fieldnames = [
        "dataset",
        "model_family",
        "model",
        "n",
        "iou_percent",
        "dice_percent",
        "precision_percent",
        "recall_percent",
        "pixel_accuracy_percent",
        "note",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Additional Cross-Dataset Evaluation Summary", ""]
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        if not subset:
            continue
        subset.sort(key=lambda row: float(row["iou_percent"]), reverse=True)
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"Note: {DATASETS[dataset]['note']}.",
                "",
                "| Model | Family | N | IoU (%) | Dice (%) | Precision (%) | Recall (%) | Pixel Accuracy (%) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in subset:
            lines.append(
                "| {model} | {model_family} | {n} | {iou_percent} | {dice_percent} | "
                "{precision_percent} | {recall_percent} | {pixel_accuracy_percent} |".format(**row)
            )
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
