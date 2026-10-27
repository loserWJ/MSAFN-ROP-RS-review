"""Load the three pre-exported 10,853-row splits; never repartition at runtime."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" if (ROOT / "data").is_dir() else ROOT / "reviewer_package_10853" / "data"
SPLIT_NAMES = ("train", "validation", "test")
RAW_SHAPES = {
    "construction": (6, 30),
    "surrounding rock": (5,),
    "mwd": (7, 75, 9),
    "out": (9,),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_datasets(folder: Path = DATASET_DIR, *, names=SPLIT_NAMES,
                        verify_hashes: bool = True):
    """Return raw tensors keyed by split and their immutable manifest."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    names = tuple(names)
    if not names or any(name not in SPLIT_NAMES for name in names):
        raise ValueError("Only train, validation and test are valid split names")
    datasets = {}
    for name in names:
        record = manifest["splits"][name]
        path = folder / record["file"]
        if verify_hashes and sha256(path) != record["sha256"]:
            raise ValueError(f"{name} data hash mismatch")
        with gzip.open(path, "rb") as stream:
            raw = torch.load(stream, map_location="cpu", weights_only=True)
        if set(raw) != set(RAW_SHAPES):
            raise ValueError(f"{name} has unexpected tensor keys")
        count = int(record["rows"])
        for key, shape in RAW_SHAPES.items():
            tensor = raw[key]
            if tuple(tensor.shape) != (count, *shape) or not torch.isfinite(tensor).all():
                raise ValueError(f"{name}.{key} has invalid shape or values")
        datasets[name] = raw
    if set(names) == set(SPLIT_NAMES) and sum(
            len(datasets[name]["out"]) for name in SPLIT_NAMES) != manifest["total_rows"]:
        raise ValueError("Split counts do not equal total rows")
    return datasets, manifest
