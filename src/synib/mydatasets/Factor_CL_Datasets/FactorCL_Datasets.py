import numpy as np
import torch

from synib.mydatasets.Factor_CL_Datasets.MultiBench.datasets.affect.get_data import get_dataloader


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
        torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0) if torch.is_tensor(m) and torch.is_floating_point(m) else m
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


class FactorCL_Dataloader:
    def __init__(self, config):
        self.config = config

        g = torch.Generator()
        seed = int(self.config.training_params.get("seed", 0))
        g.manual_seed(seed)

        data_type = self.config.dataset.get("data_type", "mosi")
        task = self.config.get("task", "classification")

        train_loader, valid_loader, test_loader = get_dataloader(
            filepath=self.config.dataset.data_roots,
            robust_test=False,
            data_type=data_type,
            task=task,
            batch_size=int(self.config.training_params.get("batch_size", 32)),
            train_shuffle=bool(self.config.dataset.get("train_shuffle", True)),
            num_workers=int(self.config.training_params.get("data_loader_workers", 2)),
            max_pad=bool(self.config.dataset.get("max_pad", False)),
            max_seq_len=int(self.config.dataset.get("max_seq_len", 50)),
            flatten_time_series=bool(self.config.dataset.get("flatten_time_series", False)),
            raw_path=self.config.dataset.get("raw_path", ""),
            z_norm=bool(self.config.dataset.get("z_norm", False)),
        )

        train_loader.generator = g
        train_loader.worker_init_fn = lambda worker_id: np.random.seed(seed + worker_id)

        self.train_loader = _BatchAdapter(train_loader, task=task)
        self.valid_loader = _BatchAdapter(valid_loader, task=task)
        self.test_loader = _BatchAdapter(test_loader, task=task)
