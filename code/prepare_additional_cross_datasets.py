import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ZIP_PREFIX = "Forest Segmented/Forest Segmented"


def ensure_clean_dirs(root):
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, mask_dir


def save_rgb(image, path):
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        pil = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    pil.save(path)


def save_mask(mask, path):
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def write_manifest(root, rows):
    with open(root / "meta_data.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "mask"])
        writer.writeheader()
        writer.writerows(rows)


def write_compat_zip(root, rows):
    zip_path = root / "mask_archive.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            zf.write(root / "images" / row["image"], f"{ZIP_PREFIX}/images/{row['image']}")
            zf.write(root / "masks" / row["mask"], f"{ZIP_PREFIX}/masks/{row['mask']}")
    return zip_path


def prepare_dead_tree(zip_path, out_root):
    image_dir, mask_dir = ensure_clean_dirs(out_root)
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        rgb_names = sorted(
            name for name in names
            if name.startswith("USA_segmentation/RGB_images/") and name.lower().endswith(".png")
        )
        for name in rgb_names:
            base = Path(name).name
            mask_name = "mask_" + base.removeprefix("RGB_")
            mask_zip_name = f"USA_segmentation/masks/{mask_name}"
            if mask_zip_name not in names:
                continue
            out_image = base
            out_mask = mask_name
            with zf.open(name) as fh:
                save_rgb(Image.open(io.BytesIO(fh.read())), image_dir / out_image)
            with zf.open(mask_zip_name) as fh:
                save_mask(Image.open(io.BytesIO(fh.read())), mask_dir / out_mask)
            rows.append({"image": out_image, "mask": out_mask})
    write_manifest(out_root, rows)
    zip_out = write_compat_zip(out_root, rows)
    print(f"dead_tree rows={len(rows)} root={out_root} zip={zip_out}", flush=True)


def prepare_amazon_hf(source_root, out_root):
    from datasets import load_dataset

    image_dir, mask_dir = ensure_clean_dirs(out_root)
    data_files = {}
    data_dir = Path(source_root) / "data"
    for split in ("train", "val"):
        files = sorted(str(p) for p in data_dir.glob(f"{split}-*.parquet"))
        if files:
            data_files[split] = files
    dataset = load_dataset("parquet", data_files=data_files)

    rows = []
    for split, ds in dataset.items():
        for idx, sample in enumerate(ds):
            filename = sample.get("filename") or f"{split}_{idx:06d}.png"
            stem = Path(str(filename)).stem
            out_image = f"{split}_{stem}.png"
            out_mask = f"{split}_{stem}_mask.png"
            save_rgb(sample["image"], image_dir / out_image)
            save_mask(sample["label"], mask_dir / out_mask)
            rows.append({"image": out_image, "mask": out_mask})
    write_manifest(out_root, rows)
    zip_out = write_compat_zip(out_root, rows)
    print(f"amazon rows={len(rows)} root={out_root} zip={zip_out}", flush=True)


def find_forest_change_masks(source_root, split):
    root = Path(source_root)
    candidates = [
        root / "masks" / split,
        root / "labels" / split,
        root / "annotations" / split,
        root / "mask" / split,
        root / "images" / split / "label",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = [p for p in root.rglob("*.png") if split in p.parts and any(x in p.parts for x in ("masks", "labels", "mask"))]
    if matches:
        return matches[0].parent
    raise FileNotFoundError(f"Could not find masks for Forest-Change split={split}")


def prepare_forest_change(source_root, out_root):
    image_dir, mask_dir = ensure_clean_dirs(out_root)
    rows = []
    root = Path(source_root)
    for split in ("train", "val", "test"):
        b_dir = root / "images" / split / "B"
        if not b_dir.exists():
            continue
        masks_dir = find_forest_change_masks(root, split)
        for image_path in sorted(b_dir.glob("*.png")):
            mask_path = masks_dir / image_path.name
            if not mask_path.exists():
                alt = image_path.name.replace(f"{split}_", "mask_")
                mask_path = masks_dir / alt
            if not mask_path.exists():
                continue
            out_image = f"{split}_B_{image_path.name}"
            out_mask = f"{split}_{mask_path.name}"
            save_rgb(Image.open(image_path), image_dir / out_image)
            save_mask(Image.open(mask_path), mask_dir / out_mask)
            rows.append({"image": out_image, "mask": out_mask})
    write_manifest(out_root, rows)
    zip_out = write_compat_zip(out_root, rows)
    print(f"forest_change rows={len(rows)} root={out_root} zip={zip_out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)

    dead = sub.add_parser("dead_tree")
    dead.add_argument("--zip-path", required=True)
    dead.add_argument("--out-root", required=True)

    amazon = sub.add_parser("amazon")
    amazon.add_argument("--source-root", required=True)
    amazon.add_argument("--out-root", required=True)

    forest = sub.add_parser("forest_change")
    forest.add_argument("--source-root", required=True)
    forest.add_argument("--out-root", required=True)

    args = parser.parse_args()
    if args.kind == "dead_tree":
        prepare_dead_tree(Path(args.zip_path), Path(args.out_root))
    elif args.kind == "amazon":
        prepare_amazon_hf(Path(args.source_root), Path(args.out_root))
    elif args.kind == "forest_change":
        prepare_forest_change(Path(args.source_root), Path(args.out_root))


if __name__ == "__main__":
    main()
