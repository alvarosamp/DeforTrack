from pathlib import Path
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


DATASET = Path(r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3")
IMAGE_DIR = DATASET / "test" / "images"
LABEL_DIR = DATASET / "test" / "labels"
MODEL = Path(r"C:\Users\vish8\Documents\Codex\2026-07-15\me\work\models\yolov8s_best.pt")
OUT_DIR = Path(r"C:\Users\vish8\Documents\Codex\2026-07-15\me\outputs\segmentation_figures")

IMG_SIZE = 640
CONF = 0.25
MAX_IMAGES = None


def read_yolo_seg_mask(label_path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        pts = np.round(coords).astype(np.int32)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask


def result_to_mask(result, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None:
        return mask
    for poly in result.masks.xy:
        pts = np.round(poly).astype(np.int32)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask


def metrics(gt: np.ndarray, pred: np.ndarray):
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = np.logical_and(gt_bool, pred_bool).sum()
    fp = np.logical_and(~gt_bool, pred_bool).sum()
    fn = np.logical_and(gt_bool, ~pred_bool).sum()
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return {
        "iou": float(iou),
        "dice": float(dice),
        "fp": int(fp),
        "fn": int(fn),
        "gt_cov": float(gt_bool.mean()),
        "pred_cov": float(pred_bool.mean()),
        "fp_rate": float(fp / gt.size),
        "fn_rate": float(fn / gt.size),
    }


def tint_mask(base_rgb: np.ndarray, mask: np.ndarray, color, alpha=0.45):
    out = base_rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_error_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    out = img.copy()
    tp = np.logical_and(gt == 1, pred == 1)
    fp = np.logical_and(gt == 0, pred == 1)
    fn = np.logical_and(gt == 1, pred == 0)
    out = tint_mask(out, tp, (40, 180, 80), 0.35)
    out = tint_mask(out, fp, (230, 60, 60), 0.55)
    out = tint_mask(out, fn, (50, 120, 255), 0.55)
    return out


def mask_panel(mask: np.ndarray, color=(60, 180, 90)):
    out = np.full((*mask.shape, 3), 245, dtype=np.uint8)
    out[mask.astype(bool)] = color
    return out


def add_label(img: Image.Image, text: str, pad=8):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = font
    box_h = 34
    draw.rectangle([0, 0, img.width, box_h], fill=(20, 20, 20))
    draw.text((pad, 7), text, fill=(255, 255, 255), font=font if len(text) < 35 else small)
    return img


PANEL = 320


def tile_for_case(case, row_title):
    img = case["image"]
    gt = case["gt"]
    pred = case["pred"]
    overlay = make_error_overlay(img, gt, pred)
    panels = [
        (img, f"{row_title}"),
        (mask_panel(gt, (70, 170, 80)), "Ground truth"),
        (mask_panel(pred, (70, 130, 230)), "Prediction"),
        (overlay, f"Errors | IoU {case['iou']*100:.1f}%"),
    ]
    pil_panels = []
    for arr, text in panels:
        im = Image.fromarray(arr).resize((PANEL, PANEL), Image.Resampling.LANCZOS)
        pil_panels.append(add_label(im, text))
    canvas = Image.new("RGB", (PANEL * 4, PANEL), (255, 255, 255))
    for i, im in enumerate(pil_panels):
        canvas.paste(im, (i * PANEL, 0))
    return canvas


def build_grid(cases, titles, out_path, header):
    row_h = PANEL
    header_h = 46
    w = PANEL * 4
    h = header_h + row_h * len(cases)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
        small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = font
    draw.rectangle([0, 0, w, header_h], fill=(250, 250, 250))
    draw.text((12, 10), header, fill=(25, 25, 25), font=font)
    for i, (case, title) in enumerate(zip(cases, titles)):
        row = tile_for_case(case, title)
        canvas.paste(row, (0, header_h + i * row_h))
    canvas.save(out_path, quality=95)


def select_correct(cases):
    candidates = [c for c in cases if c["iou"] >= 0.82 and 0.08 <= c["gt_cov"] <= 0.95]
    bins = [
        ("Fragmented vegetation", 0.08, 0.35),
        ("Forest-soil boundary", 0.35, 0.58),
        ("Dense canopy", 0.58, 0.78),
        ("High forest cover", 0.78, 0.96),
    ]
    selected = []
    titles = []
    used_prefix = set()
    for title, lo, hi in bins:
        pool = [c for c in candidates if lo <= c["gt_cov"] < hi and c["prefix"] not in used_prefix]
        if not pool:
            pool = [c for c in candidates if lo <= c["gt_cov"] < hi]
        if pool:
            chosen = sorted(pool, key=lambda x: x["iou"], reverse=True)[0]
            selected.append(chosen)
            titles.append(title)
            used_prefix.add(chosen["prefix"])
    if len(selected) < 4:
        for c in sorted(candidates, key=lambda x: x["iou"], reverse=True):
            if c not in selected:
                selected.append(c)
                titles.append("Successful segmentation")
            if len(selected) == 4:
                break
    return selected[:4], titles[:4]


def select_failures(cases):
    nontrivial = [c for c in cases if 0.05 <= c["gt_cov"] <= 0.95]
    used_family = set()

    def family(c):
        name = c["name"]
        if name.startswith("IJ_"):
            return "river_ij"
        if name.startswith("Soja"):
            return "soja"
        if name.startswith("Cafezal"):
            return "cafezal"
        if name.startswith("evo"):
            return "evo"
        if "_sat_" in name:
            return "sat"
        if name.startswith("0-"):
            return "drone_zero"
        if name.startswith("alina"):
            return "alina"
        return name.split("_")[0].split("-")[0]

    forced = {
        "False positives": "Mata_MP4-24_jpg.rf.7c2dc6630452d394a1a4f886ccc8a73d.jpg",
        "Mixed land cover": "0-22-21_png_jpg.rf.7b90667657ae39c7fc4f0bb495833775.jpg",
    }
    categories = [
        ("False positives", lambda c: c["fp_rate"] > c["fn_rate"]),
        ("False negatives", lambda c: c["fn_rate"] >= c["fp_rate"]),
        ("Ambiguous boundary", lambda c: abs(c["fp_rate"] - c["fn_rate"]) < 0.03),
        ("Mixed land cover", lambda c: True),
    ]
    selected = []
    titles = []
    for title, fn in categories:
        if title in forced:
            forced_case = next((c for c in nontrivial if c["name"] == forced[title]), None)
            if forced_case is not None:
                selected.append(forced_case)
                titles.append(title)
                used_family.add(family(forced_case))
                continue
        pool = [c for c in nontrivial if c not in selected and fn(c) and family(c) not in used_family]
        if not pool:
            pool = [c for c in nontrivial if c not in selected and fn(c)]
        if not pool:
            continue
        chosen = sorted(pool, key=lambda x: x["iou"])[0]
        selected.append(chosen)
        titles.append(title)
        used_family.add(family(chosen))
    if len(selected) < 4:
        for c in sorted(nontrivial, key=lambda x: x["iou"]):
            if c not in selected:
                selected.append(c)
                titles.append("Failure case")
            if len(selected) == 4:
                break
    return selected[:4], titles[:4]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    images = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if MAX_IMAGES:
        random.seed(7)
        images = random.sample(images, min(MAX_IMAGES, len(images)))

    cases = []
    for idx, img_path in enumerate(images, 1):
        label_path = LABEL_DIR / (img_path.stem + ".txt")
        pil = Image.open(img_path).convert("RGB")
        w, h = pil.size
        gt = read_yolo_seg_mask(label_path, w, h)
        result = model.predict(str(img_path), imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
        pred = result_to_mask(result, w, h)
        m = metrics(gt, pred)
        prefix = img_path.stem.split("_jpg")[0].split("_png")[0].split(".rf")[0]
        cases.append({
            "path": img_path,
            "name": img_path.name,
            "prefix": prefix[:16],
            "image": np.array(pil),
            "gt": gt,
            "pred": pred,
            **m,
        })
        if idx % 100 == 0:
            print(f"processed {idx}/{len(images)}")

    correct, correct_titles = select_correct(cases)
    failures, failure_titles = select_failures(cases)

    build_grid(
        correct,
        correct_titles,
        OUT_DIR / "correct_segmentation_scenarios.png",
        "Representative successful segmentations across visually distinct forest scenarios",
    )
    build_grid(
        failures,
        failure_titles,
        OUT_DIR / "segmentation_failure_cases.png",
        "Representative segmentation failure cases and error types",
    )

    summary = OUT_DIR / "selected_examples_summary.csv"
    with summary.open("w", encoding="utf-8") as f:
        f.write("figure,type,file,iou,dice,gt_coverage,pred_coverage,fp_rate,fn_rate\n")
        for fig, selected, titles in [("correct", correct, correct_titles), ("failure", failures, failure_titles)]:
            for c, title in zip(selected, titles):
                f.write(
                    f"{fig},{title},{c['name']},{c['iou']:.6f},{c['dice']:.6f},"
                    f"{c['gt_cov']:.6f},{c['pred_cov']:.6f},{c['fp_rate']:.6f},{c['fn_rate']:.6f}\n"
                )
    all_metrics = OUT_DIR / "all_test_examples_metrics.csv"
    with all_metrics.open("w", encoding="utf-8") as f:
        f.write("file,iou,dice,gt_coverage,pred_coverage,fp_rate,fn_rate\n")
        for c in sorted(cases, key=lambda x: x["iou"]):
            f.write(
                f"{c['name']},{c['iou']:.6f},{c['dice']:.6f},"
                f"{c['gt_cov']:.6f},{c['pred_cov']:.6f},{c['fp_rate']:.6f},{c['fn_rate']:.6f}\n"
            )
    print(OUT_DIR / "correct_segmentation_scenarios.png")
    print(OUT_DIR / "segmentation_failure_cases.png")
    print(summary)
    print(all_metrics)


if __name__ == "__main__":
    main()
