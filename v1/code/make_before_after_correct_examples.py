from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("models/yolov8s_best.pt")
DEFORTRACK_TEST = Path(
    r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images"
)
OUT_DIR = Path("outputs/segmentation_figures/before_after_correct")

EXAMPLES = [
    {
        "file": "before_after_high_forest_cover.png",
        "caption": "High forest-cover scenario",
        "image": "0-44-58_png_jpg.rf.2cf2d80632be1a09f2f3204ebb233d31.jpg",
    },
    {
        "file": "before_after_continuous_canopy.png",
        "caption": "Continuous canopy scenario",
        "image": "bloco_90_jpg.rf.f038d6a3cc65ef09fff3fc6f8f2f30ec.jpg",
    },
    {
        "file": "before_after_dense_green_canopy.png",
        "caption": "Dense green canopy scenario",
        "image": "IJ_08_25_33_2560_0_png_jpg.rf.3d42c7c8d240b11e832eef112ef4bd49.jpg",
    },
    {
        "file": "before_after_fragmented_canopy.png",
        "caption": "Fragmented canopy scenario",
        "image": "111335_sat_07_jpg.rf.7ff08ef32af1d069b5236700c9e59fb0.jpg",
    },
    {
        "file": "before_after_forest_soil_boundary.png",
        "caption": "Forest-soil boundary scenario",
        "image": "Soja_MP4-7_jpg.rf.8768b857fa08096bbc21a67b408674de.jpg",
    },
    {
        "file": "before_after_heterogeneous_vegetation.png",
        "caption": "Heterogeneous vegetation scenario",
        "image": "IJ_08_25_50_2560_0_png_jpg.rf.49faa2a6d970244bff8233c478513a4f.jpg",
    },
]


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((640, 640), Image.BILINEAR)
    return np.asarray(image)


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


def overlay_prediction(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    green = np.array([35, 185, 95], dtype=np.float32)
    alpha = 0.48
    overlay[mask > 0.5] = (1 - alpha) * overlay[mask > 0.5] + alpha * green
    return overlay.astype(np.uint8)


def save_before_after(model: YOLO, item: dict) -> None:
    image_path = DEFORTRACK_TEST / item["image"]
    image = load_rgb(image_path)
    mask = predict_mask(model, image_path)
    overlay = overlay_prediction(image, mask)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.4, 3.2), constrained_layout=True)
    for ax, title, panel in zip(axes, ["Before", "After"], [image, overlay]):
        ax.imshow(panel)
        ax.set_title(title, fontsize=12, weight="bold")
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
        save_before_after(model, item)


if __name__ == "__main__":
    main()
