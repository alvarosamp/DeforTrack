from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


MODEL_PATH = Path("models/yolov8s_best.pt")
DATASET_ROOT = Path(
    r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3"
)
IMAGE_DIR = DATASET_ROOT / "test" / "images"
LABEL_DIR = DATASET_ROOT / "test" / "labels"
OUT_DIR = Path("outputs/segmentation_figures/qualitative_with_masks")

EXAMPLES = [
    {
        "file": "qualitative_mask_example_01_dense_forest.png",
        "image": "IJ_08_25_33_2560_0_png_jpg.rf.3d42c7c8d240b11e832eef112ef4bd49.jpg",
        "caption": "Dense forest scenario",
    },
    {
        "file": "qualitative_mask_example_02_high_forest_cover.png",
        "image": "0-44-58_png_jpg.rf.2cf2d80632be1a09f2f3204ebb233d31.jpg",
        "caption": "High forest-cover scenario",
    },
    {
        "file": "qualitative_mask_example_03_continuous_canopy.png",
        "image": "bloco_90_jpg.rf.f038d6a3cc65ef09fff3fc6f8f2f30ec.jpg",
        "caption": "Continuous canopy scenario",
    },
    {
        "file": "qualitative_mask_example_04_forest_boundary.png",
        "image": "Soja_MP4-7_jpg.rf.8768b857fa08096bbc21a67b408674de.jpg",
        "caption": "Forest-boundary scenario",
    },
    {
        "file": "qualitative_mask_example_05_heterogeneous_vegetation.png",
        "image": "IJ_08_25_50_2560_0_png_jpg.rf.49faa2a6d970244bff8233c478513a4f.jpg",
        "caption": "Heterogeneous vegetation scenario",
    },
    {
        "file": "qualitative_mask_example_06_fragmented_canopy.png",
        "image": "111335_sat_07_jpg.rf.7ff08ef32af1d069b5236700c9e59fb0.jpg",
        "caption": "Fragmented canopy scenario",
    },
]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((640, 640), Image.BILINEAR))


def load_yolo_polygon_mask(label_path: Path, size: int = 640) -> np.ndarray:
    mask_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask_img)
    if not label_path.exists():
        return np.zeros((size, size), dtype=np.float32)

    for raw_line in label_path.read_text().splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7:
            continue
        coords = [float(value) for value in parts[1:]]
        points = [
            (coords[i] * size, coords[i + 1] * size)
            for i in range(0, len(coords) - 1, 2)
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def predict_mask(model: YOLO, path: Path) -> np.ndarray:
    result = model.predict(str(path), imgsz=640, conf=0.25, verbose=False)[0]
    mask = np.zeros((640, 640), dtype=np.float32)
    if result.masks is None:
        return mask
    for instance in result.masks.data.detach().cpu().numpy():
        instance_img = Image.fromarray((instance > 0.5).astype(np.uint8) * 255)
        instance_img = instance_img.resize((640, 640), Image.NEAREST)
        mask = np.maximum(mask, np.asarray(instance_img, dtype=np.float32) / 255.0)
    return mask


def color_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask > 0.5] = color
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    blue = np.array([34, 95, 210], dtype=np.float32)
    alpha = 0.52
    overlay[mask > 0.5] = (1 - alpha) * overlay[mask > 0.5] + alpha * blue
    return overlay.astype(np.uint8)


def save_example(model: YOLO, item: dict) -> None:
    image_path = IMAGE_DIR / item["image"]
    label_path = LABEL_DIR / (image_path.stem + ".txt")

    image = load_rgb(image_path)
    gt_mask = load_yolo_polygon_mask(label_path)
    pred_mask = predict_mask(model, image_path)
    prediction_overlay = overlay_mask(image, pred_mask)

    panels = [
        ("Input image", image),
        ("Ground-truth mask", color_mask(gt_mask, (35, 190, 95))),
        ("Predicted mask", color_mask(pred_mask, (34, 95, 210))),
        ("Prediction overlay", prediction_overlay),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.15), constrained_layout=True)
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#444444")
    fig.suptitle(item["caption"], fontsize=12, weight="bold")

    out_path = OUT_DIR / item["file"]
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(out_path.resolve())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    for item in EXAMPLES:
        save_example(model, item)


if __name__ == "__main__":
    main()
