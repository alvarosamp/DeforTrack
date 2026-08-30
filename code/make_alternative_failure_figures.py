from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


DATASET = Path(r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3")
IMAGE_DIR = DATASET / "test" / "images"
LABEL_DIR = DATASET / "test" / "labels"
MODEL = Path("models/yolov8s_best.pt")
OUT_DIR = Path("outputs/segmentation_figures/failure_alternative")
IMG_SIZE = 640
CONF = 0.25
PANEL = 320

CASES = [
    (
        "Shadow / low contrast",
        "0-49-2_png_jpg.rf.57d55aab6681f61ad87de6e6323cd058.jpg",
        "failure_shadow_low_contrast.png",
    ),
    (
        "Bright green confusion",
        "ik_4_00301_png_jpg.rf.b19594460f2540d91ff156e273db3beb.jpg",
        "failure_bright_green_confusion.png",
    ),
    (
        "Low sparse vegetation",
        "alina_110_jpg.rf.48462fd2b042e7a971956206d6a7acc3.jpg",
        "failure_low_sparse_vegetation.png",
    ),
    (
        "Subtle vegetation texture",
        "ik_4_04651_png_jpg.rf.aec57597dcf2d12a91d32385c65c8b20.jpg",
        "failure_subtle_vegetation_texture.png",
    ),
]


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


def metrics(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = np.logical_and(gt_bool, pred_bool).sum()
    fp = np.logical_and(~gt_bool, pred_bool).sum()
    fn = np.logical_and(gt_bool, ~pred_bool).sum()
    union = tp + fp + fn
    return float(tp / union) if union else 1.0


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


def add_label(img: Image.Image, text: str):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = font
    draw.rectangle([0, 0, img.width, 34], fill=(20, 20, 20))
    draw.text((8, 7), text, fill=(255, 255, 255), font=font if len(text) < 33 else small)
    return img


def tile_for_case(title: str, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, iou: float):
    overlay = make_error_overlay(image, gt, pred)
    panels = [
        (image, title),
        (mask_panel(gt, (70, 170, 80)), "Ground truth"),
        (mask_panel(pred, (70, 130, 230)), "Prediction"),
        (overlay, f"Errors | IoU {iou * 100:.1f}%"),
    ]
    pil_panels = []
    for arr, text in panels:
        im = Image.fromarray(arr).resize((PANEL, PANEL), Image.Resampling.LANCZOS)
        pil_panels.append(add_label(im, text))
    canvas = Image.new("RGB", (PANEL * 4, PANEL), (255, 255, 255))
    for i, im in enumerate(pil_panels):
        canvas.paste(im, (i * PANEL, 0))
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    rows = []
    for title, filename, out_name in CASES:
        image_path = IMAGE_DIR / filename
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_arr = np.asarray(image)
        gt = read_yolo_seg_mask(LABEL_DIR / f"{image_path.stem}.txt", width, height)
        result = model.predict(str(image_path), imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
        pred = result_to_mask(result, width, height)
        iou = metrics(gt, pred)
        row = tile_for_case(title, image_arr, gt, pred, iou)
        row.save(OUT_DIR / out_name, quality=95)
        rows.append(row)
        print((OUT_DIR / out_name).resolve())

    header_h = 46
    canvas = Image.new("RGB", (PANEL * 4, header_h + PANEL * len(rows)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, canvas.width, header_h], fill=(250, 250, 250))
    draw.text((12, 10), "Alternative representative segmentation failure cases", fill=(25, 25, 25), font=font)
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, header_h + idx * PANEL))
    out = OUT_DIR / "segmentation_failure_cases_alternative.png"
    canvas.save(out, quality=95)
    print(out.resolve())


if __name__ == "__main__":
    main()
