import csv
from pathlib import Path


base = Path(r"C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\cross_dataset")
files = [
    base / "all_images_yolo" / "cross_dataset_yolo_summary.csv",
    base / "all_images_unet" / "cross_dataset_unet_summary.csv",
    base / "all_images_detectron" / "cross_dataset_detectron_summary.csv",
]

rows = []
for file_path in files:
    with open(file_path, newline="", encoding="utf-8") as fh:
        rows.extend(list(csv.DictReader(fh)))

fields = [
    "model",
    "n",
    "iou_mean",
    "dice_mean",
    "precision_mean",
    "recall_mean",
    "pixel_accuracy_mean",
    "inference_time_s_mean",
    "iou_std",
    "dice_std",
    "precision_std",
    "recall_std",
    "pixel_accuracy_std",
    "inference_time_s_std",
]

out = base / "cross_dataset_all_models_summary.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})

print(out)
print(r"\begin{table*}[!ht]")
print(r"\centering")
print(r"\caption{Cross-dataset validation on the external forest segmentation dataset.}")
print(r"\label{tab:cross_dataset_validation}")
print(r"\begin{tabular}{lccccc}")
print(r"\hline")
print(r"Model & IoU & Dice/F1 & Precision & Recall & Pixel Acc. \\")
print(r"\hline")
for row in sorted(rows, key=lambda item: float(item["iou_mean"]), reverse=True):
    values = [
        float(row["iou_mean"]) * 100,
        float(row["dice_mean"]) * 100,
        float(row["precision_mean"]) * 100,
        float(row["recall_mean"]) * 100,
        float(row["pixel_accuracy_mean"]) * 100,
    ]
    print(
        f"{row['model']} & "
        f"{values[0]:.2f} & {values[1]:.2f} & {values[2]:.2f} & "
        f"{values[3]:.2f} & {values[4]:.2f} \\\\"
    )
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table*}")
