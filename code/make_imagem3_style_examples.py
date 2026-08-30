from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("models/yolov8s_best.pt")
DEFORTRACK_TEST = Path(
    r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images"
)
OUT_DIR = Path("outputs/segmentation_figures/imagem3_style")

FIGURES = [
    {
        "file": "imagem3_style_dense_continuous.png",
        "left": "IJ_08_25_33_2560_0_png_jpg.rf.3d42c7c8d240b11e832eef112ef4bd49.jpg",
        "right": "bloco_90_jpg.rf.f038d6a3cc65ef09fff3fc6f8f2f30ec.jpg",
    },
    {
        "file": "imagem3_style_high_fragmented.png",
        "left": "0-44-58_png_jpg.rf.2cf2d80632be1a09f2f3204ebb233d31.jpg",
        "right": "111335_sat_07_jpg.rf.7ff08ef32af1d069b5236700c9e59fb0.jpg",
    },
    {
        "file": "imagem3_style_boundary_heterogeneous.png",
        "left": "Soja_MP4-7_jpg.rf.8768b857fa08096bbc21a67b408674de.jpg",
        "right": "IJ_08_25_50_2560_0_png_jpg.rf.49faa2a6d970244bff8233c478513a4f.jpg",
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
        fontsize=18,
        family="serif",
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 4},
    )


def draw_panel_letters(fig, axes) -> None:
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for ax, letter in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], letters):
        bbox = ax.get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2,
            bbox.y0 - 0.025,
            letter,
            ha="center",
            va="top",
            fontsize=18,
            family="serif",
        )


def save_figure(model: YOLO, item: dict) -> None:
    left_path = DEFORTRACK_TEST / item["left"]
    right_path = DEFORTRACK_TEST / item["right"]
    top_left = load_rgb(left_path)
    top_right = load_rgb(right_path)
    bottom_left = predict_plot(model, left_path)
    bottom_right = predict_plot(model, right_path)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.6))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.08, wspace=0.02, hspace=0.12)

    panels = [
        (top_left, "Before"),
        (top_right, "After"),
        (bottom_left, "Before"),
        (bottom_right, "After"),
    ]

    for ax, (panel, label) in zip(axes.ravel(), panels):
        ax.imshow(panel)
        ax.axis("off")
        add_corner_label(ax, label)

    draw_panel_letters(fig, axes)

    out_path = OUT_DIR / item["file"]
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(out_path.resolve())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    for item in FIGURES:
        save_figure(model, item)


if __name__ == "__main__":
    main()
