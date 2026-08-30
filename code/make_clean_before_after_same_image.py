from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("models/yolov8s_best.pt")
DEFORTRACK_TEST = Path(
    r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images"
)
OUT_DIR = Path("outputs/segmentation_figures/before_after_same_image")

EXAMPLES = [
    {
        "file": "qualitative_example_01_dense_forest.png",
        "image": "IJ_08_25_33_2560_0_png_jpg.rf.3d42c7c8d240b11e832eef112ef4bd49.jpg",
    },
    {
        "file": "qualitative_example_02_high_forest_cover.png",
        "image": "0-44-58_png_jpg.rf.2cf2d80632be1a09f2f3204ebb233d31.jpg",
    },
    {
        "file": "qualitative_example_03_continuous_canopy.png",
        "image": "bloco_90_jpg.rf.f038d6a3cc65ef09fff3fc6f8f2f30ec.jpg",
    },
    {
        "file": "qualitative_example_04_forest_boundary.png",
        "image": "Soja_MP4-7_jpg.rf.8768b857fa08096bbc21a67b408674de.jpg",
    },
    {
        "file": "qualitative_example_05_heterogeneous_vegetation.png",
        "image": "IJ_08_25_50_2560_0_png_jpg.rf.49faa2a6d970244bff8233c478513a4f.jpg",
    },
    {
        "file": "qualitative_example_06_fragmented_canopy.png",
        "image": "111335_sat_07_jpg.rf.7ff08ef32af1d069b5236700c9e59fb0.jpg",
    },
]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((640, 420), Image.BILINEAR))


def predict_plot(model: YOLO, path: Path) -> np.ndarray:
    result = model.predict(str(path), imgsz=640, conf=0.25, verbose=False)[0]
    plotted = result.plot(labels=True, boxes=True, masks=True, line_width=2)
    plotted = Image.fromarray(plotted[..., ::-1]).convert("RGB").resize((640, 420), Image.BILINEAR)
    return np.asarray(plotted)


def add_corner_label(ax, text: str) -> None:
    ax.text(
        0.98,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=20,
        family="serif",
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 4},
    )


def save_example(model: YOLO, item: dict) -> None:
    image_path = DEFORTRACK_TEST / item["image"]
    before = load_rgb(image_path)
    after = predict_plot(model, image_path)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.16, wspace=0.025)

    for ax, panel, label in zip(axes, [before, after], ["Before", "After"]):
        ax.imshow(panel)
        ax.axis("off")
        add_corner_label(ax, label)

    for ax, letter in zip(axes, ["(a)", "(b)"]):
        bbox = ax.get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            bbox.y0 - 0.035,
            letter,
            ha="center",
            va="top",
            fontsize=18,
            family="serif",
        )

    out_path = OUT_DIR / item["file"]
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(out_path.resolve())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    for item in EXAMPLES:
        save_example(model, item)


if __name__ == "__main__":
    main()
