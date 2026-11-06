"""Input hash checks and validation-only checkpoint selection."""
import csv
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path


COUNTS = {"train": 12861, "valid": 833, "test": 892}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_rows(rows, counts=COUNTS):
    if len({(row["split"], row["image"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate manifest items")
    if any(row["split"] not in counts for row in rows):
        raise ValueError("Unknown partition")
    if {role: sum(row["split"] == role for row in rows) for role in counts} != counts:
        raise ValueError("Wrong partition counts")
    for row in rows:
        for key in ("image", "label"):
            if Path(row[key]).name != row[key] or row[key] in {".", ".."} or "/" in row[key] or "\\" in row[key]:
                raise ValueError("Manifest filenames must not escape their partition")


def verify_live_file(path, expected):
    if not Path(path).is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed training input: {path}")


def verify_partition(directory, data_root):
    directory, data_root = Path(directory), Path(data_root)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    manifest = directory / "partition_manifest.csv"
    verify_live_file(manifest, protocol["manifest_sha256"])
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_rows(rows)
    if protocol["counts"] != COUNTS or any(row["pairs_at_or_above_threshold"] for row in protocol["comparisons"]):
        raise ValueError("Unapproved partition protocol")
    for role in COUNTS:
        expected = [str(data_root / role / "images" / row["image"]) for row in rows if row["split"] == role]
        actual = (directory / f"{role}.txt").read_text(encoding="utf-8").splitlines()
        if [str(Path(path).resolve()) for path in actual] != [str(Path(path).resolve()) for path in expected]:
            raise ValueError(f"Changed {role} image list")
    for index, row in enumerate(rows, 1):
        verify_live_file(data_root / row["split"] / "images" / row["image"], row["image_sha256"])
        verify_live_file(data_root / row["split"] / "labels" / row["label"], row["label_sha256"])
        if index % 1000 == 0:
            print(f"Training preflight hashes {index}/{len(rows)}", flush=True)
    return protocol


def validate_selection(job):
    if job["selection_split"] != "valid" or job["selection_images"] != 833 or job["test_used_for_selection"]:
        raise ValueError("Test partition must not select the checkpoint")


def validate_weight_entry(entry):
    if entry["initialization_type"] != "official COCO pretrained" or not entry["source_url"].startswith("https://github.com/ultralytics/assets/releases/download/"):
        raise ValueError("Missing official initial-weight provenance")
    verify_live_file(entry["path"], entry["sha256"])


def require_free_gpu_pipeline(status_path):
    state = json.loads(Path(status_path).read_text(encoding="utf-8"))
    if state["phase"] != "ready_for_manuscript_review":
        raise RuntimeError("Existing GPU training/evaluation pipeline has not completed")


@contextmanager
def gpu_training_lease(path, pipeline_status):
    require_free_gpu_pipeline(pipeline_status)
    path = Path(path)
    token = uuid.uuid4().hex
    owner = dict(pid=os.getpid(), token=token)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Another fresh experiment owns the GPU lease; investigate stale leases, never overwrite") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(owner, handle)
    try:
        yield
    finally:
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != owner:
            raise RuntimeError("GPU lease ownership changed; refusing to remove another owner's lease")
        path.unlink()
