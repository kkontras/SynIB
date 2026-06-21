"""
Generate 10-fold cross-validation splits for FactorCL datasets.

Saves {dataset}_kfold10.pkl alongside the original pkl with structure:
    {
        "pooled": {
            "text": np.ndarray, "vision": np.ndarray, "audio": np.ndarray,
            "labels": np.ndarray, "id": ...
        },
        "folds": {
            "0": {"train": [idx, ...], "valid": [idx, ...], "test": [idx, ...]},
            ...
            "9": {...}
        }
    }

For MOSI: splitting is done at the VIDEO level (all utterances from the
same video go to the same fold) to prevent leakage across the train/test boundary.

For MUStARD/URFunny: stratified sample-level splitting.

Val assignment: fold i → test, fold (i+1)%10 → val, rest → train.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 10
RANDOM_STATE = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def drop_entry(dataset):
    """Drop samples where all text features are zero (matches dataloader logic)."""
    drop = set()
    for i, k in enumerate(dataset["text"]):
        if np.asarray(k).sum() == 0:
            drop.add(i)
    if not drop:
        return dataset
    keep = [i for i in range(len(dataset["text"])) if i not in drop]
    out = {}
    for key, val in dataset.items():
        if isinstance(val, np.ndarray):
            out[key] = val[keep]
        elif isinstance(val, list):
            out[key] = [val[i] for i in keep]
        else:
            out[key] = val
    return out


def pool_splits(data):
    """Concatenate train / valid / test into a single pooled dict."""
    splits = ["train", "valid", "test"]
    keys = list(data["train"].keys())
    pooled = {}
    for k in keys:
        parts = [data[s][k] for s in splits]
        if all(isinstance(p, np.ndarray) for p in parts):
            pooled[k] = np.concatenate(parts, axis=0)
        else:
            result = []
            for p in parts:
                if isinstance(p, list):
                    result.extend(p)
                else:
                    result.extend(p.tolist())
            pooled[k] = result
    return pooled


def binarize(labels, data_type):
    """Return binary {0,1} array for stratification purposes."""
    arr = np.asarray(labels).flatten().astype(float)
    if data_type == "mosi":
        return (arr > 0).astype(int)
    else:
        # mustard: labels are -1/1; ur_funny: labels are 0/1
        return (arr > 0).astype(int)


def extract_video_id(id_val):
    """Extract video-level ID from utterance ID (MOSI: 'videoId_uttIdx')."""
    if isinstance(id_val, bytes):
        id_val = id_val.decode()
    s = str(id_val).strip()
    return s.rsplit("_", 1)[0]


# ── core fold builder ─────────────────────────────────────────────────────────

def build_folds(pooled, data_type, video_level=False):
    n = (pooled["labels"].shape[0] if isinstance(pooled["labels"], np.ndarray)
         else len(pooled["labels"]))
    labels_bin = binarize(pooled["labels"], data_type)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    sample_fold = np.full(n, -1, dtype=int)

    if video_level:
        ids = pooled["id"]
        if isinstance(ids, np.ndarray) and ids.ndim > 1:
            video_ids = [extract_video_id(row[0]) for row in ids]
        elif isinstance(ids, np.ndarray):
            video_ids = [extract_video_id(x) for x in ids]
        else:
            video_ids = [extract_video_id(x) for x in ids]

        unique_vids = sorted(set(video_ids))
        vid2idx = {v: i for i, v in enumerate(unique_vids)}
        n_vids = len(unique_vids)

        # Majority-vote label per video
        vid_pos = np.zeros(n_vids); vid_cnt = np.zeros(n_vids)
        for i, vid in enumerate(video_ids):
            j = vid2idx[vid]; vid_pos[j] += labels_bin[i]; vid_cnt[j] += 1
        vid_label = (vid_pos / np.maximum(vid_cnt, 1) >= 0.5).astype(int)

        vid_fold = np.full(n_vids, -1, dtype=int)
        for fi, (_, test_vi) in enumerate(skf.split(np.zeros(n_vids), vid_label)):
            vid_fold[test_vi] = fi

        for i, vid in enumerate(video_ids):
            sample_fold[i] = vid_fold[vid2idx[vid]]
    else:
        for fi, (_, test_si) in enumerate(skf.split(np.zeros(n), labels_bin)):
            sample_fold[test_si] = fi

    folds = {}
    for fi in range(N_SPLITS):
        val_fi = (fi + 1) % N_SPLITS
        test_mask  = sample_fold == fi
        val_mask   = sample_fold == val_fi
        train_mask = ~test_mask & ~val_mask
        folds[str(fi)] = {
            "train": np.where(train_mask)[0].tolist(),
            "valid": np.where(val_mask)[0].tolist(),
            "test":  np.where(test_mask)[0].tolist(),
        }
        n_trn = train_mask.sum(); n_val = val_mask.sum(); n_tst = test_mask.sum()
        pos_tst = labels_bin[test_mask].mean()
        print(f"  fold {fi}: train={n_trn}  val={n_val}  test={n_tst}  "
              f"test_pos_rate={pos_tst:.2f}")

    return folds


# ── per-dataset entry points ──────────────────────────────────────────────────

DATASETS = {
    "mustard": {
        "pkl":        "prepared/mustard/mustard_data.pkl",
        "out":        "prepared/mustard/mustard_kfold10.pkl",
        "data_type":  "mustard",
        "video_level": False,
    },
    "mosi": {
        "pkl":        "prepared/mosi/mosi_data.pkl",
        "out":        "prepared/mosi/mosi_kfold10.pkl",
        "data_type":  "mosi",
        "video_level": True,
    },
    "ur_funny": {
        "pkl":        "prepared/ur_funny/ur_funny_data.pkl",
        "out":        "prepared/ur_funny/ur_funny_kfold10.pkl",
        "data_type":  "humor",
        "video_level": False,
    },
}


def process_dataset(name, cfg, base_dir):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    pkl_path = base_dir / cfg["pkl"]
    out_path = base_dir / cfg["out"]

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("  Applying drop_entry …")
    for split in ("train", "valid", "test"):
        before = (len(data[split]["text"])
                  if isinstance(data[split]["text"], list)
                  else data[split]["text"].shape[0])
        data[split] = drop_entry(data[split])
        after = (len(data[split]["text"])
                 if isinstance(data[split]["text"], list)
                 else data[split]["text"].shape[0])
        print(f"    {split}: {before} → {after} samples")

    print("  Pooling splits …")
    pooled = pool_splits(data)
    n_total = (pooled["labels"].shape[0] if isinstance(pooled["labels"], np.ndarray)
               else len(pooled["labels"]))
    print(f"  Total pooled samples: {n_total}")

    print("  Building folds …")
    folds = build_folds(pooled, cfg["data_type"], cfg["video_level"])

    result = {"pooled": pooled, "folds": folds}
    with open(out_path, "wb") as f:
        pickle.dump(result, f, protocol=4)
    print(f"  Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate 10-fold CV splits for FactorCL datasets.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()),
                    help="Which datasets to process (default: all).")
    args = ap.parse_args()

    base_dir = Path(__file__).parent
    for name in args.datasets:
        process_dataset(name, DATASETS[name], base_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
