"""Profile one trained model on all 892 DeforTrack test images.

The complete test set is read once before measurement to warm the operating
system file cache without retaining decoded images in process memory. Reported
latency then covers image decoding, model-specific preprocessing, and inference
under a consistent warm-cache condition. CPU is normalized to the total
logical-processor capacity, RAM is the process peak RSS, and GPU energy is
integrated and normalized as J/image.
"""

import argparse
import csv
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import psutil
import torch


ARCH_TO_CONFIG = {
    "mask_rcnn_R_101_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    "mask_rcnn_R_101_DC5_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "mask_rcnn_R_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "mask_rcnn_R_50_C4_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
    "mask_rcnn_R_50_C4_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
    "mask_rcnn_R_50_DC5_1x": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
}


class ResourceSampler:
    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.logical_cpus = psutil.cpu_count(logical=True) or 1
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.nvml = None
        self.gpu = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None

    def _sample(self):
        now = time.perf_counter()
        cpu = self.process.cpu_percent(interval=None) / self.logical_cpus
        ram = self.process.memory_info().rss / (1024 * 1024)
        power = np.nan
        if self.nvml is not None:
            try:
                power = self.nvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000.0
            except Exception:
                power = np.nan
        self.samples.append((now, cpu, ram, power))

    def _run(self):
        self.process.cpu_percent(interval=None)
        self._sample()
        while not self.stop_event.wait(self.interval):
            self._sample()
        self._sample()

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass

    def summary(self, n_images: int) -> dict:
        arr = np.asarray(self.samples, dtype=float)
        times = arr[:, 0] - arr[0, 0]
        powers = arr[:, 3]
        valid = np.isfinite(powers)
        energy = float(np.trapezoid(powers[valid], times[valid])) if valid.sum() > 1 else float("nan")
        return {
            "ram_peak_mb": float(arr[:, 2].max()),
            "cpu_mean_percent_total": float(arr[:, 1].mean()),
            "gpu_power_mean_w": float(np.nanmean(powers)) if valid.any() else float("nan"),
            "energy_total_j": energy,
            "energy_j_per_image": energy / n_images if n_images and np.isfinite(energy) else float("nan"),
            "resource_samples": len(self.samples),
        }


def build_detectron(model_name: str, weights: str):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(ARCH_TO_CONFIG[model_name]))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.30
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 100
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 20
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 640
    return DefaultPredictor(cfg)


def synchronize_cuda() -> None:
    """Wait for queued GPU work so elapsed time includes completed inference."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["yolo", "detectron", "unet"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=892)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    image_paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise RuntimeError("No test images found")

    if args.family == "yolo":
        from ultralytics import YOLO

        model = YOLO(args.model_path)

        def infer(path):
            image = cv2.imread(str(path))
            model.predict(image, imgsz=640, conf=0.25, device=0, verbose=False)

    elif args.family == "detectron":
        model = build_detectron(args.model_name, args.model_path)

        def infer(path):
            image = cv2.imread(str(path))
            model(image)

    else:
        import keras

        model = keras.saving.load_model(args.model_path, compile=False)

        def infer(path):
            image = cv2.imread(str(path))
            image = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2RGB)
            model.predict(np.expand_dims(image.astype(np.float32) / 255.0, axis=0), verbose=0)

    # Prime the OS file cache equally for every model, without retaining the
    # decoded dataset and inflating process peak RAM during measurement.
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not decode test image: {path}")
    del image

    if args.warmup != len(image_paths):
        raise ValueError("The standardized protocol requires one full test-set warm-up pass")
    for path in image_paths:
        infer(path)
        synchronize_cuda()

    baseline_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    trial_elapsed = []
    for repetition in range(1, args.repetitions + 1):
        trial_started = time.perf_counter()
        for index, path in enumerate(image_paths, start=1):
            infer(path)
            synchronize_cuda()
            if index % 100 == 0:
                print(
                    f"{args.model_name} trial {repetition}/{args.repetitions}: "
                    f"{index}/{len(image_paths)}",
                    flush=True,
                )
        trial_elapsed.append(time.perf_counter() - trial_started)
    elapsed = time.perf_counter() - started
    sampler.stop()

    measured_inferences = len(image_paths) * args.repetitions

    result = {
        "model": args.model_name,
        "family": args.family,
        "images": len(image_paths),
        "warmup_images": args.warmup,
        "repetitions": args.repetitions,
        "measured_inferences": measured_inferences,
        "model_size_mb": Path(args.model_path).stat().st_size / (1024 * 1024),
        "elapsed_s": elapsed,
        "trial_elapsed_s_json": json.dumps(trial_elapsed),
        "trial_latency_s_mean": float(np.mean(np.asarray(trial_elapsed) / len(image_paths))),
        "trial_latency_s_std": float(np.std(np.asarray(trial_elapsed) / len(image_paths), ddof=1)),
        "inference_time_s_per_image": elapsed / measured_inferences,
        "fps": measured_inferences / elapsed,
        "ram_baseline_mb": baseline_ram,
    }
    result.update(sampler.summary(measured_inferences))
    result["ram_peak_mb"] = max(result["ram_peak_mb"], baseline_ram)
    result["ram_delta_mb"] = result["ram_peak_mb"] - baseline_ram

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
