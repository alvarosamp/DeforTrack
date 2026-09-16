"""Build public uncertainty and external Top-3 tables from validated summaries."""
import csv, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"consolidated"; OUT.mkdir(parents=True,exist_ok=True)

def read(path): return list(csv.DictReader(path.open(encoding="utf-8")))

sources=[
    ("YOLO",ROOT/"results/internal_test_892/yolo_test_summary.csv"),
    ("Mask R-CNN",ROOT/"results/internal_test_892/maskrcnn_test_summary.csv"),
    ("U-Net",ROOT/"results/internal_test_892/unet_892_summary.csv"),
]
metrics=[("IoU","iou"),("Dice/F1","dice_f1"),("Precision","precision"),("Recall","recall"),("Boundary IoU","boundary_iou"),("Boundary F1","boundary_f1"),("Pixel accuracy","pixel_accuracy")]
rows=[]
for family,path in sources:
    for r in read(path):
        for label,key in metrics:
            mean=float(r[f"{key}_mean"]); std=float(r[f"{key}_std"]); n=int(r["n"])
            ci=float(r.get(f"{key}_ci95_halfwidth") or 1.96*std/math.sqrt(n))
            rows.append({"family":family,"model":r["model"],"n":n,"metric":label,"mean_percent":100*mean,"std_percent":100*std,"ci95_halfwidth_pp":100*ci,"ci95_lower_percent":100*(mean-ci),"ci95_upper_percent":100*(mean+ci)})
with (OUT/"internal_uncertainty_all_models.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

external=read(ROOT/"results/cross_dataset/additional_cross_dataset_summary.csv")
top=[]
for dataset in ("Forest Aerial Images","Amazon/Atlantic Forest"):
    candidates=[r for r in external if r["dataset"]==dataset]
    for rank,r in enumerate(sorted(candidates,key=lambda x:float(x["iou_percent"]),reverse=True)[:3],1):
        top.append({"dataset":dataset,"rank":rank,"model":r["model"],"family":r["model_family"],"n":r["n"],"iou_percent":r["iou_percent"],"dice_percent":r["dice_percent"],"precision_percent":r["precision_percent"],"recall_percent":r["recall_percent"],"pixel_accuracy_percent":r["pixel_accuracy_percent"]})
with (OUT/"external_top3.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=top[0]); w.writeheader(); w.writerows(top)
print(f"Wrote {len(rows)} uncertainty rows and {len(top)} external Top-3 rows")
