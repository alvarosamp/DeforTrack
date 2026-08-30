import csv
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_DIR = Path(r"D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\test\images")
METRICS_CSV = Path("outputs/segmentation_figures/all_test_examples_metrics.csv")
OUT = Path("outputs/segmentation_figures/failure_candidates_contact_sheet.jpg")


def main() -> None:
    with METRICS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [
        r
        for r in rows
        if float(r["iou"]) < 0.55 and 0.02 < float(r["gt_coverage"]) < 0.80
    ]
    rows = sorted(rows, key=lambda r: float(r["iou"]))[:72]

    thumb_w, thumb_h = 160, 160
    label_h = 34
    cols = 6
    rows_count = 12
    sheet = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(rows):
        img_path = IMAGE_DIR / row["file"]
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + thumb_h), f"{i}: IoU {float(row['iou']) * 100:.1f}", fill=(0, 0, 0))
        draw.text((x + 2, y + thumb_h + 15), row["file"][:24], fill=(0, 0, 0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=92)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
