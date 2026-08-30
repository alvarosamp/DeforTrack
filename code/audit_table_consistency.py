"""Verify that manuscript table values match the consolidated CSV outputs."""

import argparse
import csv
import re
from pathlib import Path


def read_csv(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table_block(tex: str, label: str) -> str:
    label_index = tex.index(rf"\label{{{label}}}")
    start = tex.rfind(r"\begin{table", 0, label_index)
    end = tex.index(r"\end{table", label_index)
    return tex[start:end]


def clean(cell: str) -> str:
    cell = cell.replace(r"\textbf{", "").replace(r"\mathbf{", "")
    return cell.replace("{", "").replace("}", "").replace("$", "").strip()


def model_rows(block: str, model_names: set[str]) -> dict[str, list[str]]:
    rows = {}
    for line in block.splitlines():
        if "&" not in line or r"\hline" not in line:
            continue
        cells = [clean(cell) for cell in line.split(r"\\")[0].split("&")]
        if cells and cells[0] in model_names:
            rows[cells[0]] = cells
    return rows


def same_number(actual: str, expected: str | float, tolerance: float = 0.0051) -> None:
    if expected == "NA":
        if actual != "--":
            raise ValueError(f"Expected --, found {actual}")
        return
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"Expected {expected}, found {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--computational", required=True)
    args = parser.parse_args()

    tex = Path(args.tex).read_text(encoding="utf-8")
    segmentation = read_csv(args.segmentation)
    computational = read_csv(args.computational)
    names = {row["model"] for row in segmentation}

    seg_rows = model_rows(table_block(tex, "tab:segmentation_metrics"), names)
    comp_rows = model_rows(table_block(tex, "tab:computational_metrics"), names)
    if set(seg_rows) != names or set(comp_rows) != names:
        raise ValueError("One or more model rows are missing from a manuscript table")

    seg_fields = [
        "images", "iou_percent", "map_50_percent", "map_50_95_percent",
        "dice_f1_percent", "boundary_iou_percent", "boundary_f1_percent",
        "pixel_accuracy_percent", "precision_percent", "recall_percent",
    ]
    for expected in segmentation:
        actual = seg_rows[expected["model"]]
        for index, field in enumerate(seg_fields, start=1):
            same_number(actual[index], expected[field])

    comp_fields = ["images", "model_size_mb", "peak_process_ram_mb"]
    for expected in computational:
        actual = comp_rows[expected["model"]]
        for index, field in enumerate(comp_fields, start=1):
            same_number(actual[index], expected[field])
        latency = re.findall(r"\d+\.\d+", actual[4])
        if len(latency) != 2:
            raise ValueError(f"Invalid latency cell for {expected['model']}: {actual[4]}")
        same_number(latency[0], expected["latency_s_per_image"], tolerance=0.000051)
        same_number(latency[1], expected["trial_latency_s_std"], tolerance=0.000051)
        same_number(actual[5], expected["fps"])
        same_number(actual[6], expected["normalized_cpu_percent"])
        same_number(actual[7], expected["gpu_energy_j_per_image"], tolerance=0.00051)

    print("Manuscript tables match all 17 consolidated segmentation and computational rows.")


if __name__ == "__main__":
    main()
