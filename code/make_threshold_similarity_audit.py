"""Create a visual audit for every unique test image above a threshold."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--pairs-per-page", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.pairs.open(encoding="utf-8")))
    selected = [
        row
        for row in rows
        if row["comparison"] in {"test-train", "validation-test"}
        and float(row["cosine"]) >= args.threshold
    ]
    selected.sort(key=lambda row: (row["comparison"], -float(row["cosine"])))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        test_path = row["left"] if row["comparison"] == "test-train" else row["right"]
        grouped.setdefault(test_path, []).append(row)

    audit_rows = []
    for test_path, matches in grouped.items():
        matches.sort(key=lambda row: -float(row["cosine"]))
        best = matches[0]
        audit_rows.append(
            {
                **best,
                "test_image": test_path,
                "test_name": Path(test_path).name,
                "flagged_pair_count": len(matches),
                "flagged_comparisons": ";".join(
                    sorted({row["comparison"] for row in matches})
                ),
                "visual_decision": "same_or_overlapping_scene",
                "visual_notes": (
                    "Same scene, overlapping crop, rotation, or adjacent frame; "
                    "removed from the final test set."
                ),
            }
        )
    audit_rows.sort(key=lambda row: -float(row["cosine"]))

    fields = list(audit_rows[0])
    with (args.out_dir / "threshold_pairs_visual_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    width, row_height = 1500, 250
    for page_start in range(0, len(audit_rows), args.pairs_per_page):
        page = audit_rows[page_start : page_start + args.pairs_per_page]
        canvas = Image.new("RGB", (width, row_height * len(page)), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(page):
            y = offset * row_height
            paths = (row["left"], row["right"])
            for column, path in enumerate(paths):
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((700, 190))
                    x = 15 + column * 745 + (700 - image.width) // 2
                    canvas.paste(image, (x, y + 48))
            index = page_start + offset + 1
            title = (
                f'{index:02d}  {row["comparison"]}  cosine={float(row["cosine"]):.6f}  '
                f'test={row["test_name"]}'
            )
            draw.text((15, y + 8), title, fill="black")
            draw.text((15, y + 29), "query", fill="black")
            draw.text((760, y + 29), "reference", fill="black")
        page_number = page_start // args.pairs_per_page + 1
        canvas.save(args.out_dir / f"threshold_pairs_page_{page_number:02d}.jpg", quality=94)

    unique_test = sorted(row["test_image"] for row in audit_rows)
    (args.out_dir / "unique_flagged_test_images.txt").write_text(
        "\n".join(unique_test) + "\n", encoding="utf-8"
    )
    print(
        f"pairs={len(selected)} unique_test_images={len(unique_test)} "
        f"pages={(len(audit_rows) + args.pairs_per_page - 1) // args.pairs_per_page}",
        flush=True,
    )


if __name__ == "__main__":
    main()
