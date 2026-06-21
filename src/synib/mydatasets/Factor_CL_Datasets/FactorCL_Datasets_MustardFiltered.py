import numpy as np
import torch
from torch.utils.data import DataLoader

from .MultiBench.datasets.affect.get_data import (
    Affectdataset,
    _process_1,
    _process_2,
    drop_entry,
)
import pickle


def _convert_affect_batch(batch, task="classification"):
    """Convert MultiBench affect batch tuples to the project batch schema."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 4:
        return batch

    sample_idx = None
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) >= 3:
        modalities = batch[0]
        if len(batch) > 2 and torch.is_tensor(batch[2]):
            sample_idx = batch[2]
        label = batch[3]
    elif all(torch.is_tensor(batch[i]) for i in [0, 1, 2]):
        modalities = [batch[0], batch[1], batch[2]]
        if len(batch) > 2 and torch.is_tensor(batch[2]):
            sample_idx = batch[2]
        label = batch[3]
    else:
        return batch

    # Keep values finite so a single corrupted sample cannot poison training.
    modalities = [
        torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.is_tensor(m) and torch.is_floating_point(m)
        else m
        for m in modalities
    ]

    if isinstance(label, torch.Tensor) and label.ndim > 1 and label.shape[-1] == 1:
        label = label.squeeze(-1)

    if isinstance(label, torch.Tensor):
        if torch.is_floating_point(label):
            label = torch.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
        if task == "classification":
            label = label.long()
            label = torch.where(label < 0, torch.zeros_like(label), label)
        else:
            label = label.float()

    out = {
        "data": {
            "c": modalities[0],
            "f": modalities[1],
            "g": modalities[2],
        },
        "label": label,
    }
    if sample_idx is not None:
        out["sample_idx"] = sample_idx.view(-1)
    return out


class _BatchAdapter:
    """Wrap a DataLoader and adapt each batch on iteration."""

    def __init__(self, loader, task="classification"):
        self.loader = loader
        self.task = task
        self.dataset = loader.dataset

    def __iter__(self):
        for batch in self.loader:
            yield _convert_affect_batch(batch, task=self.task)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, item):
        return getattr(self.loader, item)


def _first_text_nonzero_timestep(text_sample):
    nz = np.argwhere(text_sample != 0)
    if nz.size == 0:
        return None
    return int(nz[0][0])


def _filter_corrupt_mustard_samples(dataset_split, split_name, cfg):
    """Drop samples with extreme/corrupt video features on the aligned segment."""
    threshold = float(cfg.get("vision_absmax_threshold", 1e6))
    enabled = bool(cfg.get("enabled", True))
    check_aligned_segment = bool(cfg.get("check_aligned_segment", True))
    min_aligned_len = int(cfg.get("min_aligned_len", 1))

    if not enabled:
        print(f"[MustardFilteredDataloader] {split_name}: corrupt-video filter disabled")
        return dataset_split

    n = int(dataset_split["vision"].shape[0])
    keep_idx = []
    dropped = []

    for i in range(n):
        text_i = dataset_split["text"][i]
        start = 0
        if check_aligned_segment:
            start = _first_text_nonzero_timestep(text_i)
            if start is None:
                dropped.append((i, "text_empty"))
                continue

        vision_i = dataset_split["vision"][i][start:]
        aligned_len = int(dataset_split["vision"][i].shape[0] - start)
        if aligned_len < min_aligned_len or vision_i.size == 0:
            dropped.append((i, f"short_aligned_len<{min_aligned_len}"))
            continue

        if not np.isfinite(vision_i).all():
            dropped.append((i, "vision_non_finite"))
            continue

        v_absmax = float(np.max(np.abs(vision_i)))
        if v_absmax > threshold:
            dropped.append((i, f"vision_absmax>{threshold:g} ({v_absmax:.3g})"))
            continue

        keep_idx.append(i)

    if len(keep_idx) == n:
        print(f"[MustardFilteredDataloader] {split_name}: no corrupt video samples detected")
        return dataset_split

    filtered = {}
    for k, v in dataset_split.items():
        if isinstance(v, np.ndarray):
            filtered[k] = v[keep_idx]
        else:
            filtered[k] = [v[i] for i in keep_idx]

    print(
        f"[MustardFilteredDataloader] {split_name}: dropped {len(dropped)}/{n} samples "
        f"(vision absmax threshold={threshold:g}, aligned={check_aligned_segment})"
    )
    if dropped:
        preview = ", ".join(f"{idx}:{reason}" for idx, reason in dropped[:10])
        if len(dropped) > 10:
            preview += ", ..."
        print(f"[MustardFilteredDataloader] {split_name}: first drops -> {preview}")

    return filtered


def _load_affect_with_optional_mustard_filter(config):
    filepath = config.dataset.data_roots
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    # Preserve baseline text-empty filtering from MultiBench affect loader.
    alldata["train"] = drop_entry(alldata["train"])
    alldata["valid"] = drop_entry(alldata["valid"])
    alldata["test"] = drop_entry(alldata["test"])

    data_type = str(config.dataset.get("data_type", "mosi")).lower()
    filter_cfg = config.dataset.get("mustard_video_filter", {})
    if data_type in {"mustard", "sarcasm"}:
        for split in ("train", "valid", "test"):
            alldata[split] = _filter_corrupt_mustard_samples(alldata[split], split, filter_cfg)

    return alldata


class FactorCL_MustardFiltered_Dataloader:
    """
    FactorCL affect dataloader variant with explicit Mustard video corruption filtering.

    Keeps the implementation separate so the filtering policy is easy to inspect/change.
    """

    def __init__(self, config):
        self.config = config

        g = torch.Generator()
        seed = int(self.config.training_params.get("seed", 0))
        g.manual_seed(seed)

        data_type = self.config.dataset.get("data_type", "mosi")
        task = self.config.get("task", "classification")
        batch_size = int(self.config.training_params.get("batch_size", 32))
        num_workers = int(self.config.training_params.get("data_loader_workers", 2))
        train_shuffle = bool(self.config.dataset.get("train_shuffle", True))
        max_pad = bool(self.config.dataset.get("max_pad", False))
        max_seq_len = int(self.config.dataset.get("max_seq_len", 50))
        flatten_time_series = bool(self.config.dataset.get("flatten_time_series", False))
        z_norm = bool(self.config.dataset.get("z_norm", False))

        process = _process_2 if max_pad else _process_1

        fold = self.config.dataset.get("fold", None)
        use_kfold = (fold is not None and
                     self.config.dataset.get("kfold_data_roots", None) is not None)

        if use_kfold:
            kfold_path = self.config.dataset.kfold_data_roots
            with open(kfold_path, "rb") as _f:
                kfold = pickle.load(_f)
            pooled     = kfold["pooled"]
            fold_entry = kfold["folds"][str(int(fold))]

            def _subset(pool, indices):
                out = {}
                for k, v in pool.items():
                    if isinstance(v, np.ndarray):
                        out[k] = v[indices]
                    else:
                        out[k] = [v[i] for i in indices]
                return out

            processed_dataset = {
                "train": _subset(pooled, fold_entry["train"]),
                "valid": _subset(pooled, fold_entry["valid"]),
                "test":  _subset(pooled, fold_entry["test"]),
            }
            # Apply Mustard corruption filter to kfold splits too
            filter_cfg = self.config.dataset.get("mustard_video_filter", {})
            if data_type in {"mustard", "sarcasm"}:
                for split in ("train", "valid", "test"):
                    processed_dataset[split] = _filter_corrupt_mustard_samples(
                        processed_dataset[split], split, filter_cfg)
        else:
            processed_dataset = _load_affect_with_optional_mustard_filter(self.config)

        train_loader = DataLoader(
            Affectdataset(
                processed_dataset["train"],
                flatten_time_series,
                task=task,
                max_pad=max_pad,
                max_pad_num=max_seq_len,
                data_type=data_type,
                z_norm=z_norm,
            ),
            shuffle=train_shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=process,
        )

        valid_loader = DataLoader(
            Affectdataset(
                processed_dataset["valid"],
                flatten_time_series,
                task=task,
                max_pad=max_pad,
                max_pad_num=max_seq_len,
                data_type=data_type,
                z_norm=z_norm,
            ),
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=process,
        )

        test_loader = DataLoader(
            Affectdataset(
                processed_dataset["test"],
                flatten_time_series,
                task=task,
                max_pad=max_pad,
                max_pad_num=max_seq_len,
                data_type=data_type,
                z_norm=z_norm,
            ),
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=process,
        )

        train_loader.generator = g
        train_loader.worker_init_fn = lambda worker_id: np.random.seed(seed + worker_id)

        self.train_loader = _BatchAdapter(train_loader, task=task)
        self.valid_loader = _BatchAdapter(valid_loader, task=task)
        self.test_loader = _BatchAdapter(test_loader, task=task)
