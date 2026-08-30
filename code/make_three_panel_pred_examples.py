from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("models/yolov8s_best.pt")
IMAGE_DIR = Path(
    r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images"
)
OUT_DIR = Path("outputs/segmentation_figures/qualitative_three_panel")

EXAMPLES = [
    {
        "file": "qualitative_pred_example_01_dense_forest.png",
        "title": "Dense forest scenario",
        "image": "IJ_08_25_33_2560_0_png_jpg.rf.3d42c7c8d240b11e832eef112ef4bd49.jpg",
    },
    {
        "file": "qualitative_pred_example_02_high_forest_cover.png",
        "title": "High forest-cover scenario",
        "image": "0-44-58_png_jpg.rf.2cf2d80632be1a09f2f3204ebb233d31.jpg",
    },
    {
        "file": "qualitative_pred_example_03_continuous_canopy.png",
        "title": "Continuous canopy scenario",
        "image": "bloco_90_jpg.rf.f038d6a3cc65ef09fff3fc6f8f2f30ec.jpg",
    },
    {
        "file": "qualitative_pred_example_04_forest_boundary.png",
        "title": "Forest-boundary scenario",
        "image": "Soja_MP4-7_jpg.rf.8768b857fa08096bbc21a67b408674de.jpg",
    },
    {
        "file": "qualitative_pred_example_05_heterogeneous_vegetation.png",
        "title": "Heterogeneous vegetation scenario",
        "image": "IJ_08_25_50_2560_0_png_jpg.rf.49faa2a6d970244bff8233c478513a4f.jpg",
    },
    {
        "file": "qualitative_pred_example_06_fragmented_canopy.png",
        "title": "Fragmented canopy scenario",
        "image": "111335_sat_07_jpg.rf.7ff08ef32af1d069b5236700c9e59fb0.jpg",
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
    green = np.array([38, 190, 96], dtype=np.float32)
    alpha = 0.48
    overlay[mask > 0.5] = (1 - alpha) * overlay[mask > 0.5] + alpha * green
    return overlay.astype(np.uint8)


def save_example(model: YOLO, item: dict) -> None:
    image_path = IMAGE_DIR / item["image"]
    image = load_rgb(image_path)
    mask = predict_mask(model, image_path)
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
