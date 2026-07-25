"""Empirical train-split class prior for the rebuttal reference ablation.

Computed once at startup and fixed thereafter. For CREMA-D-Irony the dataset's
`label` list already reflects the α-mutated train split (irony class included),
so the fast path is exact per α.
"""

import torch


def compute_class_prior(train_loader, num_classes):
    """Return (prior, counts) over the train split's served labels.

    Fast path: dataset exposes a materialized `label` list (IronyCremadDataset).
    Fallback: iterate the loader once and collect served `label` tensors — this
    uses the exact pipeline the trainer sees (FactorCL collate, HM memmap).
    """
    labels = None
    ds = getattr(train_loader, "dataset", None)
    lab = getattr(ds, "label", None) if ds is not None else None
    if isinstance(lab, (list, tuple)) and len(lab) > 0 and isinstance(lab[0], (int,)):
        labels = torch.as_tensor(lab, dtype=torch.long)

    if labels is None:
        parts = []
        for batch in train_loader:
            if isinstance(batch, dict) and "label" in batch:
                parts.append(batch["label"].detach().cpu().flatten().long())
        if not parts:
            raise RuntimeError("compute_class_prior: could not extract labels from the train loader")
        labels = torch.cat(parts)

    counts = torch.bincount(labels.clamp_min(0), minlength=int(num_classes)).float()
    prior = counts / counts.sum()
    return prior, counts
