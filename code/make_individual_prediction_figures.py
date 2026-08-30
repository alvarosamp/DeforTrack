from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("models/yolov8s_best.pt")
OUT_DIR = Path("outputs/segmentation_figures/individual")

EXAMPLES = [
    {
        "file": "defortrack_vegetation_field_boundary.png",
        "title": "DeforTrack: vegetation-field boundary",
        "path": Path(
            r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images\0-8-33_png_jpg.rf.4557720141184f10cd9eff22ac750fb5.jpg"
        ),
    },
    {
        "file": "defortrack_dense_canopy.png",
        "title": "DeforTrack: dense canopy",
        "path": Path(
            r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images\IJ_08_25_34_2560_0_png_jpg.rf.7d5ddee0f98b1394c77cbd3e2f469e24.jpg"
        ),
    },
    {
        "file": "forest_aerial_forest_field_boundary.png",
        "title": "External Forest Aerial: forest-field boundary",
        "path": Path(
            r"D:\Datasets\Defortrack\Forest Segmented\Forest Segmented\images\10452_sat_08.jpg"
        ),
    },
    {
        "file": "forest_aerial_mixed_land_cover.png",
        "title": "External Forest Aerial: mixed land cover",
        "path": Path(
            r"D:\Datasets\Defortrack\Forest Segmented\Forest Segmented\images\111335_sat_75.jpg"
        ),
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
    data = result.masks.data.detach().cpu().numpy()
    for instance in data:
        instance_img = Image.fromarray((instance > 0.5).astype(np.uint8) * 255)
        instance_img = instance_img.resize((640, 640), Image.NEAREST)
        mask = np.maximum(mask, np.asarray(instance_img, dtype=np.float32) / 255.0)
    return mask


def overlay_prediction(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    green = np.array([38, 190, 96], dtype=np.float32)
    alpha = 0.48
    overlay[mask > 0.5] = (1 - alpha) * overlay[mask > 0.5] + alpha * green
    return overlay.astype(np.uint8)


def save_example(model: YOLO, item: dict) -> None:
    image = load_rgb(item["path"])
    mask = predict_mask(model, item["path"])
    overlay = overlay_prediction(image, mask)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.0, 3.0), constrained_layout=True)
    panels = [
        ("Input image", image),
        ("Predicted forest mask", mask),
        ("Prediction overlay", overlay),
    ]
    for ax, (title, panel) in zip(axes, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#444444")
    fig.suptitle(item["title"], fontsize=12, weight="bold")
    fig.savefig(OUT_DIR / item["file"], dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    for item in EXAMPLES:
        save_example(model, item)
        print((OUT_DIR / item["file"]).resolve())


if __name__ == "__main__":
    main()
