"""
Raw Hateful Memes Dataset + Dataloader (image + text + label).

Mirrors the key contract of ESNLI_VE_ClassificationDataset so the rest of the
training pipeline needs no changes.

__getitem__ returns: {"id": str, "text": str, "image": (3,H,W) float tensor, "label": long tensor}
collate emits:       {"data": {0: [text_list], 1: stacked_images}, "id": [...], "label": ...}
"""

from __future__ import annotations

import logging
import multiprocessing
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .hm_utils import (
    LABELLED_SPLITS,
    SPLITS,
    HMMissingDataError,
    has_labels,
    load_hm_jsonl,
    record_label,
    resolve_image_path,
    verify_hm_layout,
)


class HatefulMemes_ClassificationDataset(Dataset):
    def __init__(
        self,
        config,
        split: str = "train",
        image_size: int = 224,
        image_subdir: Optional[str] = None,
        drop_invalid_labels: bool = True,
    ):
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"Unknown HM split {split!r}. Expected one of {SPLITS}.")
        self.logger = logging.getLogger("HatefulMemes")
        self.split = split

        self.data_root = str(config.dataset.data_roots)
        subdir = image_subdir or getattr(config.dataset, "image_subdir", "img_clean")
        self.paths = verify_hm_layout(self.data_root, image_subdir=subdir)
        self.image_dir = self.paths["image_dir"]

        raw = load_hm_jsonl(self.data_root, split)
        self.records: List[Dict[str, Any]] = []
        skipped = 0
        for rec in raw:
            lab = record_label(rec)
            if has_labels(split):
                if lab is None and drop_invalid_labels:
                    skipped += 1
                    continue
            self.records.append({
                "id": str(rec.get("id")),
                "text": str(rec.get("text", "")).strip(),
                "label": -1 if lab is None else int(lab),
                "img_rel": rec.get("img", ""),
            })
        if skipped:
            self.logger.info(f"{split}: skipped {skipped} records with missing labels.")
        self.logger.info(f"{split}: kept {len(self.records)} records.")

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = resolve_image_path({"img": rec["img_rel"]}, self.image_dir)
        with Image.open(img_path) as im:
            image = self.tf(im.convert("RGB"))
        return {
            "id": rec["id"],
            "text": rec["text"],
            "image": image,
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
        }


def collate_hm(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "data": {
            0: [b["text"] for b in batch],
            1: torch.stack([b["image"] for b in batch], dim=0),
        },
        "id": [b["id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch], dim=0),
    }


class HatefulMemes_Dataloader:
    """Train/valid/test splits pulled from train / dev_unseen / test_seen by default.

    Picks dev_unseen as the valid loader (primary model-selection signal per the plan).
    Test loader is test_seen (public labels).
    """

    def __init__(
        self,
        config,
        *,
        valid_split: str = "dev_unseen",
        test_split: str = "test_seen",
        image_size: int = 224,
        image_subdir: Optional[str] = None,
    ):
        self.logger = logging.getLogger("HatefulMemes DataLoader")
        batch_size = int(config.training_params.batch_size)
        test_batch_size = int(getattr(config.training_params, "test_batch_size", batch_size))

        g = torch.Generator()
        g.manual_seed(int(getattr(config.training_params, "seed", 0)))

        def seed_worker(_worker_id):
            seed = torch.initial_seed() % 2**32
            np.random.seed(seed)
            random.seed(seed)

        total_cpus = multiprocessing.cpu_count()
        workers = int(getattr(config.training_params, "data_loader_workers", 0))
        if workers < 0:
            workers = max(1, min(24, total_cpus - 1))

        kwargs = dict(
            image_size=image_size,
            image_subdir=image_subdir,
        )

        train_ds = HatefulMemes_ClassificationDataset(config, split="train", **kwargs)
        valid_ds = HatefulMemes_ClassificationDataset(config, split=valid_split, **kwargs)
        test_ds = HatefulMemes_ClassificationDataset(config, split=test_split, **kwargs)

        self.collate_fn = collate_hm

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            collate_fn=self.collate_fn,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )
        self.valid_loader = DataLoader(
            valid_ds,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
