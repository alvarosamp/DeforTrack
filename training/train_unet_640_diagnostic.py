"""U-Net definition and 640x640 training loop."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTHONHASHSEED", "0")

import cv2
import keras
import numpy as np


def rasterize(label_path: Path, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 7:
                continue
            coords = np.asarray(values[1:], dtype=np.float32)
            if coords.size % 2:
                continue
            points = coords.reshape(-1, 2)
            points[:, 0] = np.clip(np.rint(points[:, 0] * (size - 1)), 0, size - 1)
            points[:, 1] = np.clip(np.rint(points[:, 1] * (size - 1)), 0, size - 1)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask[..., None].astype(np.float32)


class SegmentationDataset(keras.utils.PyDataset):
    def __init__(self, image_dir: Path, label_dir: Path, batch_size: int, size: int, shuffle: bool, limit: int | None = None):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.paths = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if limit is not None:
            rng = np.random.default_rng(0)
            selected = np.sort(rng.choice(len(self.paths), size=min(limit, len(self.paths)), replace=False))
            self.paths = [self.paths[index] for index in selected]
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.size = size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        return (len(self.paths) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        images, masks = [], []
        for item in ids:
            path = self.paths[item]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[:2] != (self.size, self.size):
                image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            images.append(image.astype(np.float32) / 255.0)
            masks.append(rasterize(self.label_dir / f"{path.stem}.txt", self.size))
        return np.stack(images), np.stack(masks)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.default_rng(0).shuffle(self.indices)


def build_unet(size: int) -> keras.Model:
    inputs = keras.Input((size, size, 3), name="input_layer")
    c1 = keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="conv2d")(inputs)
    c1 = keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="conv2d_1")(c1)
    p1 = keras.layers.MaxPooling2D(name="max_pooling2d")(c1)
    c2 = keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="conv2d_2")(p1)
    c2 = keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="conv2d_3")(c2)
    p2 = keras.layers.MaxPooling2D(name="max_pooling2d_1")(c2)
    c3 = keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="conv2d_4")(p2)
    c3 = keras.layers.Conv2D(256, 3, activation="relu", padding="same", name="conv2d_5")(c3)
    u2 = keras.layers.UpSampling2D(name="up_sampling2d")(c3)
    u2 = keras.layers.Concatenate(name="concatenate")([u2, c2])
    c4 = keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="conv2d_6")(u2)
    c4 = keras.layers.Conv2D(128, 3, activation="relu", padding="same", name="conv2d_7")(c4)
    u1 = keras.layers.UpSampling2D(name="up_sampling2d_1")(c4)
    u1 = keras.layers.Concatenate(name="concatenate_1")([u1, c1])
    c5 = keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="conv2d_8")(u1)
    c5 = keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="conv2d_9")(c5)
    outputs = keras.layers.Conv2D(
        1, 1, activation="sigmoid", dtype="float32", name="conv2d_10"
    )(c5)
    return keras.Model(inputs, outputs, name="unet_640_diagnostic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--transfer-only", action="store_true")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    keras.utils.set_random_seed(0)
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = keras.saving.load_model(args.source_model, compile=False)
    model = build_unet(args.size)
    transferred = []
    for layer in model.layers:
        try:
            source_layer = source.get_layer(layer.name)
        except ValueError:
            continue
        if source_layer.get_weights():
            layer.set_weights(source_layer.get_weights())
            transferred.append(layer.name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    train = SegmentationDataset(
        args.dataset_root / "train" / "images",
        args.dataset_root / "train" / "labels",
        args.batch_size,
        args.size,
        True,
        args.max_train,
    )
    validation = SegmentationDataset(
        args.dataset_root / "test" / "images",
        args.dataset_root / "test" / "labels",
        args.batch_size,
        args.size,
        False,
        args.max_validation,
    )
    protocol = {
        "purpose": "resolution-controlled U-Net diagnostic",
        "source_model": str(args.source_model),
        "input_size": args.size,
        "source_input_size": 256,
        "transferred_layers": transferred,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "mixed_precision": args.mixed_precision,
        "loss": "binary_crossentropy",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": 0,
        "training_images": len(train.paths),
        "validation_images": len(validation.paths),
        "validation_directory_role": "895-image validation partition (export directory test)",
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    if args.transfer_only:
        model.save(args.out_dir / "unet_640_transferred.keras")
        return

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.out_dir / "unet_640_best.keras", monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(args.out_dir / "history.csv"),
    ]
    history = model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    model.save(args.out_dir / "unet_640_final.keras")
    (args.out_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
