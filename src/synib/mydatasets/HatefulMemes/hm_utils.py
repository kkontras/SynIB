"""
Helpers for the Hateful Memes dataset.

Data must be placed manually at `config.dataset.data_roots` in the MMF layout:

    data_roots/
        img/            (original memes with text baked in)
        img_clean/      (inpainted variant from Fine-Grained Hateful Memes)
        train.jsonl
        dev_seen.jsonl
        dev_unseen.jsonl
        test_seen.jsonl
        test_unseen.jsonl

Sources:
    - Challenge release: https://www.drivendata.org/competitions/70/hateful-memes-phase-2/
    - Fine-Grained release (contains img_clean): https://github.com/facebookresearch/fine_grained_hateful_memes
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional


SPLITS = ("train", "dev_seen", "dev_unseen", "test_seen", "test_unseen")
LABELLED_SPLITS = ("train", "dev_seen", "dev_unseen", "test_seen")
LABEL2IDX = {0: 0, 1: 1, "0": 0, "1": 1, False: 0, True: 1}


class HMMissingDataError(FileNotFoundError):
    pass


def _jsonl_path(data_root: str, split: str) -> str:
    return os.path.join(data_root, f"{split}.jsonl")


def verify_hm_layout(data_root: str, image_subdir: str = "img_clean") -> Dict[str, str]:
    """Confirm that the expected directory layout exists. Returns a dict of paths.

    image_subdir:
        "img_clean" — inpainted variant (recommended).
        "img"       — original (text baked in).
    """
    if not os.path.isdir(data_root):
        raise HMMissingDataError(
            f"Hateful Memes data_root not found: {data_root!r}. "
            "Download the challenge data from DrivenData and the inpainted "
            "images from facebookresearch/fine_grained_hateful_memes."
        )

    img_dir = os.path.join(data_root, image_subdir)
    if not os.path.isdir(img_dir):
        raise HMMissingDataError(
            f"Missing image directory {img_dir!r}. Expected subdir {image_subdir!r} "
            f"under {data_root!r}. If you only have the original images, set "
            f"image_subdir='img' (this leaks text via OCR)."
        )

    missing = [s for s in SPLITS if not os.path.isfile(_jsonl_path(data_root, s))]
    if missing:
        raise HMMissingDataError(
            f"Missing jsonl split(s) under {data_root!r}: {missing}. "
            "All five HM splits must be present (test_unseen has no public labels)."
        )

    return {
        "data_root": data_root,
        "image_dir": img_dir,
        **{s: _jsonl_path(data_root, s) for s in SPLITS},
    }


def load_hm_jsonl(data_root: str, split: str) -> List[Dict[str, Any]]:
    """Load one HM split as a list of dicts. Does not validate image presence."""
    path = _jsonl_path(data_root, split)
    if not os.path.isfile(path):
        raise HMMissingDataError(f"Missing {path!r}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_hm_jsonl(data_root: str, split: str) -> Iterator[Dict[str, Any]]:
    path = _jsonl_path(data_root, split)
    if not os.path.isfile(path):
        raise HMMissingDataError(f"Missing {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_image_path(record: Dict[str, Any], image_dir: str) -> str:
    """HM jsonl 'img' field looks like 'img/12345.png' — strip the prefix and join with our image_dir."""
    rel = str(record.get("img", "")).strip()
    if not rel:
        raise ValueError(f"Record missing 'img' field: {record}")
    name = os.path.basename(rel)
    return os.path.join(image_dir, name)


def record_label(record: Dict[str, Any]) -> Optional[int]:
    """Return the binary label for a record, or None if missing (test_unseen)."""
    if "label" not in record or record["label"] is None:
        return None
    raw = record["label"]
    if raw in LABEL2IDX:
        return int(LABEL2IDX[raw])
    try:
        return int(raw)
    except Exception:
        return None


def has_labels(split: str) -> bool:
    return split in LABELLED_SPLITS
