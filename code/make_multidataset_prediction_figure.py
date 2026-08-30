from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("models/yolov8s_best.pt")
OUT_DIR = Path("outputs/segmentation_figures")
OUT_PATH = OUT_DIR / "model_predictions_multidataset.png"

EXAMPLES = [
    {
        "title": "DeforTrack: vegetation-field boundary",
        "path": Path(
            r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images\0-8-33_png_jpg.rf.4557720141184f10cd9eff22ac750fb5.jpg"
        ),
    },
    {
        "title": "DeforTrack: dense canopy",
        "path": Path(
            r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images\IJ_08_25_34_2560_0_png_jpg.rf.7d5ddee0f98b1394c77cbd3e2f469e24.jpg"
        ),
    },
    {
        "title": "Forest Aerial: forest-field boundary",
        "path": Path(
            r"D:\Datasets\Defortrack\Forest Segmented\Forest Segmented\images\10452_sat_08.jpg"
        ),
    },
    {
        "title": "Forest Aerial: mixed land cover",
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


def main() -> None:
    missing = [str(item["path"]) for item in EXAMPLES if not item["path"].exists()]
    if missing:
        raise FileNotFoundError("Missing input images:\n" + "\n".join(missing))
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))

    fig, axes = plt.subplots(
        nrows=len(EXAMPLES),
        ncols=3,
        figsize=(8.8, 9.7),
        constrained_layout=True,
    )
    headers = ["Input image", "Predicted forest mask", "Prediction overlay"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=11, weight="bold")

    for row, item in enumerate(EXAMPLES):
        image = load_rgb(item["path"])
        mask = predict_mask(model, item["path"])
        overlay = overlay_prediction(image, mask)

        axes[row, 0].imshow(image)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(overlay)

        axes[row, 0].text(
            0.02,
            0.98,
            item["title"],
            transform=axes[row, 0].transAxes,
            fontsize=8.4,
            weight="bold",
            color="white",
            va="top",
            ha="left",
            bbox={
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.62,
                "pad": 3.0,
            },
        )
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#444444")

    fig.suptitle(
        "YOLOv8s predictions on DeforTrack-like RGB forest imagery",
        fontsize=13,
        weight="bold",
    )
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
