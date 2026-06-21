"""
Build {dataset}_synergy3.pkl — a kfold-format pickle whose folds 0/1/2 all
hold the SAME Synergy train/valid/test partition. Running --fold 0|1|2
against this file thus gives three seeds on one fixed split (seeds 109/19/337
from train.py), byte-identical to the protocol that produced the reported
baseline numbers.

Strategy:
  - Pooling in generate_kfold_splits.py concatenates splits in order
    [train, valid, test]. So after applying drop_entry, Synergy's indices
    on the pooled array are trivial cumulative ranges.
  - We reuse the already-materialised pooled arrays from {dataset}_kfold10.pkl
    and just rewrite the `folds` key.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from synib.mydatasets.Factor_CL_Datasets.generate_kfold_splits import (
    drop_entry,
    DATASETS,
)


RAW_SOURCES = {
    "mosi":     "./data/FactorCL_Raw/raw_sources/mosi/mosi_data.pkl",
    "mustard":  "./data/FactorCL_Raw/raw_sources/mustard/sarcasm.pkl",
    "ur_funny": "./data/FactorCL_Raw/raw_sources/ur_funny/humor.pkl",
}


def _split_size_after_drop(split_dict):
    kept = drop_entry(split_dict)
    t = kept["text"]
    return t.shape[0] if isinstance(t, np.ndarray) else len(t)


def _pooled_len(pooled):
    t = pooled["text"]
    return t.shape[0] if isinstance(t, np.ndarray) else len(t)


def process_dataset(name, base_dir, raw_path, out_name="synergy3"):
    cfg = DATASETS[name]
    kfold10_path = base_dir / cfg["out"]
    out_path = kfold10_path.with_name(f"{name}_{out_name}.pkl")

    print(f"\n[{name}]")
    print(f"  raw:     {raw_path}")
    print(f"  kfold10: {kfold10_path}")
    print(f"  out:     {out_path}")

    with open(raw_path, "rb") as f:
        raw = pickle.load(f)
    n_train = _split_size_after_drop(raw["train"])
    n_valid = _split_size_after_drop(raw["valid"])
    n_test  = _split_size_after_drop(raw["test"])
    total_synergy = n_train + n_valid + n_test
    print(f"  synergy (post-drop): train={n_train}  valid={n_valid}  test={n_test}  total={total_synergy}")

    with open(kfold10_path, "rb") as f:
        kfold10 = pickle.load(f)
    pooled = kfold10["pooled"]
    n_pool = _pooled_len(pooled)
    print(f"  pooled: n={n_pool}")

    if n_pool != total_synergy:
        raise RuntimeError(
            f"{name}: pooled len ({n_pool}) != synergy total ({total_synergy}). "
            "Pooling order or drop_entry mismatch — stop before writing."
        )

    train_idx = list(range(0, n_train))
    valid_idx = list(range(n_train, n_train + n_valid))
    test_idx  = list(range(n_train + n_valid, total_synergy))

    split = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    folds = {"0": split, "1": split, "2": split}

    result = {"pooled": pooled, "folds": folds}
    with open(out_path, "wb") as f:
        pickle.dump(result, f, protocol=4)
    print(f"  saved → {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(RAW_SOURCES.keys()),
                    choices=list(RAW_SOURCES.keys()))
    ap.add_argument("--out-name", default="synergy3",
                    help="Suffix for the output pickle ({name}_{out_name}.pkl).")
    args = ap.parse_args()

    base_dir = Path(__file__).parent
    for name in args.datasets:
        process_dataset(name, base_dir, RAW_SOURCES[name], args.out_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
