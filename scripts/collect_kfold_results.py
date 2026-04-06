"""
Collect 10-fold cross-validation results for JointTF vs JointTF_IHA
across Mustard, UR-Funny, MOSEI, and MOSI.

For each (dataset, method) pair:
  - Reads the .args file to know expected jobs (fold × nlayers)
  - Checks which checkpoints exist on disk
  - Loads each checkpoint, extracts val_acc and test_acc
  - For each fold, picks the best nlayer by val_acc
  - Reports completion status and mean±std test accuracy table

Run from repo root:
    python scripts/collect_kfold_results.py
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Experiment registry ────────────────────────────────────────────────────
# Each entry: (display_name, checkpoint_base_dir, args_file)
EXPERIMENTS = [
    {
        "dataset":  "Mustard",
        "method":   "JointTF",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mustard/vt"),
        "args_file": os.path.join(REPO_ROOT, "run/condor/mustard_joint_tf_kfold.args"),
    },
    {
        "dataset":  "Mustard",
        "method":   "JointTF_IHA",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mustard/vt"),
        "args_file": os.path.join(REPO_ROOT, "run/condor/mustard_joint_tf_iha_kfold.args"),
    },
    {
        "dataset":  "UR-Funny",
        "method":   "JointTF",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/ur_funny/vt"),
        "args_file": os.path.join(REPO_ROOT, "run/condor/urfunny_joint_tf_kfold.args"),
    },
    {
        "dataset":  "UR-Funny",
        "method":   "JointTF_IHA",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/ur_funny/vt"),
        "args_file": os.path.join(REPO_ROOT, "run/condor/urfunny_joint_tf_iha_kfold.args"),
    },
    {
        "dataset":  "MOSEI",
        "method":   "JointTF",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mosei/vt"),
        "args_file": os.path.join(REPO_ROOT, "run/condor/mosei_joint_tf_kfold.args"),
    },
    {
        "dataset":  "MOSEI",
        "method":   "JointTF_IHA",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mosei/vt"),
        "args_file": os.path.join(REPO_ROOT, "run/condor/mosei_joint_tf_iha_kfold.args"),
    },
    {
        "dataset":  "MOSI",
        "method":   "JointTF",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mosi/vt"),
        "args_file": os.path.join(REPO_ROOT, "run/condor/mosi_joint_tf_kfold.args"),
    },
    {
        "dataset":  "MOSI",
        "method":   "JointTF_IHA",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mosi/vt"),
        "args_file": os.path.join(REPO_ROOT, "run/condor/mosi_joint_tf_iha_kfold.args"),
    },
]


# ─── Helpers ────────────────────────────────────────────────────────────────

def parse_args_line(line: str) -> dict:
    """Parse a single args-file line into a dict of named parameters."""
    tokens = line.split()
    params = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            # Boolean flags that take no value
            if key in ("no_model_save", "start_over", "pre", "frozen"):
                params[key] = True
                i += 1
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                params[key] = tokens[i + 1]
                i += 2
            else:
                params[key] = True
                i += 1
        else:
            i += 1
    return params


def build_suffix(p: dict) -> str:
    """
    Reconstruct the checkpoint suffix `m` using the same ordering as
    train.py / show.py.
    """
    m = ""
    if p.get("fold") is not None:
        m += "fold{}".format(p["fold"])
    if p.get("validate_with") is not None:
        m += "_vld{}".format(p["validate_with"])
    if p.get("lr") is not None:
        m += "_lr{}".format(p["lr"])
    if p.get("wd") is not None:
        m += "_wd{}".format(p["wd"])
    if p.get("batch_size") is not None:
        m += "_bs{}".format(p["batch_size"])
    # IHA params
    if p.get("pseudo_heads_q") is not None:
        m += "_phq{}".format(p["pseudo_heads_q"])
    if p.get("pseudo_heads_kv") is not None:
        m += "_phkv{}".format(p["pseudo_heads_kv"])
    if p.get("iha_init") is not None:
        m += "_ihainit{}".format(p["iha_init"])
    if p.get("iha_lr") is not None:
        m += "_ihalr{}".format(p["iha_lr"])
    if p.get("iha_layers") is not None:
        m += "_ihaL{}".format(p["iha_layers"].replace(",", "-"))
    # num_layers (added by train.py, after IHA params)
    if p.get("num_layers") is not None:
        m += "_nlayers{}".format(p["num_layers"])
    return m


def model_name_from_config(config_path: str) -> str:
    """Derive the model prefix (e.g. 'JointTF' or 'JointTF_IHA') from config."""
    import json
    full = os.path.join(REPO_ROOT, config_path)
    with open(full) as f:
        cfg = json.load(f)
    save_dir = cfg.get("model", {}).get("save_dir", "{}.pth.tar")
    # save_dir looks like "JointTF_{}.pth.tar"
    prefix = save_dir.split("_{}")[0]
    return prefix


def extract_metrics(ckpt_path: str):
    """
    Load a checkpoint and return (val_acc, test_acc) for the 'combined' head.
    Returns (None, None) on failure.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [WARN] Could not load {os.path.basename(ckpt_path)}: {e}")
        return None, None

    try:
        best_logs = ckpt["logs"]["best_logs"]
        # Key is "best_v{validate_with}" — we use "best_vaccuracy"
        val_metrics = best_logs.get("best_vaccuracy", best_logs)

        val_acc = val_metrics.get("acc", {}).get("combined", None)
        step = val_metrics.get("step", None)

        test_acc = None
        if step is not None and "test_logs" in ckpt["logs"]:
            test_logs = ckpt["logs"]["test_logs"]
            if step in test_logs:
                t = test_logs[step]
                # Some older checkpoints prefix keys with "test_"
                if "acc" not in t and "test_acc" in t:
                    t = {k.replace("test_", ""): v for k, v in t.items()}
                test_acc = t.get("acc", {}).get("combined", None)

        return val_acc, test_acc
    except Exception as e:
        print(f"  [WARN] Metric extraction failed for {os.path.basename(ckpt_path)}: {e}")
        return None, None


# ─── Main ────────────────────────────────────────────────────────────────────

def process_experiment(exp: dict):
    """
    Parse .args file, check completeness, load metrics.
    Returns:
        fold_results: dict[fold_int -> dict[nlayers_int -> {"val": float, "test": float}]]
        missing: dict[nlayers_int -> list[int]]  (folds missing per nlayers)
    """
    args_file = exp["args_file"]
    ckpt_dir = exp["ckpt_dir"]

    if not os.path.exists(args_file):
        print(f"  [ERROR] .args file not found: {args_file}")
        return {}, {}

    with open(args_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Parse each line → group by (fold, nlayers)
    model_prefix = None
    jobs = {}  # (fold, nlayers) -> ckpt_path

    for line in lines:
        p = parse_args_line(line)
        fold = int(p["fold"])
        nlayers = int(p.get("num_layers", 1))

        if model_prefix is None:
            model_prefix = model_name_from_config(p["config"])

        suffix = build_suffix(p)
        filename = f"{model_prefix}_{suffix}.pth.tar"
        ckpt_path = os.path.join(ckpt_dir, filename)
        jobs[(fold, nlayers)] = ckpt_path

    all_nlayers = sorted(set(nl for (_, nl) in jobs))
    all_folds = sorted(set(f for (f, _) in jobs))

    # fold_results[fold][nlayers] = {"val": float, "test": float}
    fold_results = defaultdict(dict)
    missing = defaultdict(list)  # nlayers -> list of missing folds

    for (fold, nlayers), ckpt_path in sorted(jobs.items()):
        if not os.path.exists(ckpt_path):
            missing[nlayers].append(fold)
            continue
        val_acc, test_acc = extract_metrics(ckpt_path)
        if val_acc is None or test_acc is None:
            missing[nlayers].append(fold)
        else:
            fold_results[fold][nlayers] = {"val": val_acc, "test": test_acc}

    return dict(fold_results), dict(missing), all_nlayers, all_folds


def main():
    # Accumulate results keyed by (dataset, method)
    results  = {}   # (dataset, method) -> fold_results
    missing  = {}   # (dataset, method) -> {nlayers -> [missing folds]}
    nlayers_map = {}  # (dataset, method) -> all_nlayers list

    for exp in EXPERIMENTS:
        key = (exp["dataset"], exp["method"])
        print(f"\nProcessing {exp['dataset']} / {exp['method']} ...")
        fold_results, miss, all_nl, all_folds = process_experiment(exp)
        results[key] = fold_results
        missing[key] = miss
        nlayers_map[key] = all_nl

    all_nl_global = [1, 2, 4]

    # ── Completion status table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPLETION STATUS (10-fold CV, per nlayers)")
    print("=" * 80)
    header = f"{'Dataset':<12}  {'Method':<14}  " + "  ".join(f"nl={n} done" for n in all_nl_global)
    print(header)
    print("-" * 80)
    for exp in EXPERIMENTS:
        key = (exp["dataset"], exp["method"])
        fold_res = results[key]
        miss = missing[key]
        row = f"{exp['dataset']:<12}  {exp['method']:<14}"
        for nl in all_nl_global:
            done = sum(1 for f in range(10) if nl in fold_res.get(f, {}))
            missing_folds = miss.get(nl, [])
            row += f"  {done}/10"
            if missing_folds:
                row += f" (miss:{sorted(missing_folds)})"
            else:
                row += "          "
        print(row)

    # ── Results table: per nlayers ───────────────────────────────────────────
    datasets_order = ["Mustard", "UR-Funny", "MOSEI", "MOSI"]
    methods = ["JointTF", "JointTF_IHA"]

    for method in methods:
        print("\n" + "=" * 80)
        print(f"TEST ACCURACY — {method}  (mean ± std, 10-fold CV)")
        print("=" * 80)
        col_w = 20
        header = f"{'Dataset':<12}" + "".join(f"  {'nlayers='+str(nl):^{col_w}}" for nl in all_nl_global)
        print(header)
        print("-" * 80)
        for dataset in datasets_order:
            key = (dataset, method)
            if key not in results:
                print(f"{dataset:<12}  N/A")
                continue
            fold_res = results[key]
            row = f"{dataset:<12}"
            for nl in all_nl_global:
                test_accs = [fold_res[f][nl]["test"] for f in range(10)
                             if f in fold_res and nl in fold_res[f]]
                if not test_accs:
                    row += f"  {'no data':^{col_w}}"
                else:
                    mean = np.mean(test_accs) * 100
                    std  = np.std(test_accs)  * 100
                    cell = f"{mean:.1f} ± {std:.1f} (n={len(test_accs)})"
                    row += f"  {cell:^{col_w}}"
            print(row)

    # ── Per-fold detail ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-FOLD DETAIL")
    print("=" * 80)
    for exp in EXPERIMENTS:
        key = (exp["dataset"], exp["method"])
        fold_res = results[key]
        if not fold_res:
            continue
        print(f"\n{exp['dataset']} / {exp['method']}")
        header = f"  {'Fold':>5}" + "".join(f"  {'nl='+str(nl)+' val%':>12}  {'test%':>6}" for nl in all_nl_global)
        print(header)
        for fold in range(10):
            row = f"  {fold:>5}"
            for nl in all_nl_global:
                if fold in fold_res and nl in fold_res[fold]:
                    r = fold_res[fold][nl]
                    row += f"  {r['val']*100:>12.2f}  {r['test']*100:>6.2f}"
                else:
                    row += f"  {'--':>12}  {'--':>6}"
            print(row)


if __name__ == "__main__":
    main()
