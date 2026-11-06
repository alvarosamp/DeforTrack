"""Export portable replacement-test artifacts without local absolute paths."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PATH_COLUMNS = {
    "left", "right", "test_image", "query", "reference",
    "removed_test_image", "removed_test_label", "replacement_image",
    "replacement_mask", "replacement_nearest_existing_path",
    "candidate_image", "candidate_mask", "nearest_existing_path",
    "output_image", "output_label", "source_image", "source_mask_or_label",
    "test_image", "reference_image",
}


def portable_csv(source: Path, destination: Path) -> None:
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    for row in rows:
        for column in PATH_COLUMNS & row.keys():
            row[column] = Path(row[column]).name if row[column] else ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_files = {
        args.root / "visual_audit" / "threshold_pairs_visual_audit.csv":
            args.out / "visual_audit" / "threshold_pairs_visual_audit.csv",
        args.root / "selection" / "replacements_manifest.csv":
            args.out / "selection" / "replacements_manifest.csv",
        args.root / "selection" / "candidate_similarity_audit.csv":
            args.out / "selection" / "candidate_similarity_audit.csv",
        args.root / "final_audit" / "final_test_nearest_similarity.csv":
            args.out / "final_audit" / "final_test_nearest_similarity.csv",
        args.test_root / "test_set_manifest.csv":
            args.out / "test_set_manifest.csv",
    }
    for source, destination in csv_files.items():
        portable_csv(source, destination)

    copies = [
        (args.root / "final_audit" / "final_similarity_audit.json",
         args.out / "final_audit" / "final_similarity_audit.json"),
        (args.test_root / "build_summary.json", args.out / "build_summary.json"),
    ]
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for page in sorted((args.root / "visual_audit").glob("threshold_pairs_page_*.jpg")):
        destination = args.out / "visual_audit" / page.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(page, destination)
    print(f"Portable artifacts written to {args.out}")


if __name__ == "__main__":
    main()
