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
        "args_files": [os.path.join(REPO_ROOT, "run/condor/mustard_joint_tf_kfold_v2.args")],
    },
    {
        "dataset":  "Mustard",
        "method":   "JointTF_IHA",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mustard/vt"),
        "args_files": [os.path.join(REPO_ROOT, "run/condor/mustard_joint_tf_iha_kfold_v2.args")],
    },
    {
        "dataset":  "UR-Funny",
        "method":   "JointTF",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/ur_funny/vt"),
        "args_files": [
            os.path.join(REPO_ROOT, "run/condor/urfunny_joint_tf_kfold.args"),
            os.path.join(REPO_ROOT, "run/condor/urfunny_joint_tf_nl8.args"),
        ],
    },
    {
        "dataset":  "UR-Funny",
        "method":   "JointTF_IHA",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/ur_funny/vt"),
        "args_files": [
            os.path.join(REPO_ROOT, "run/condor/urfunny_joint_tf_iha_kfold_v2.args"),
            os.path.join(REPO_ROOT, "run/condor/urfunny_joint_tf_iha_nl8.args"),
        ],
    },
    {
        "dataset":  "MOSEI",
        "method":   "JointTF",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mosei/vt"),
        "args_files": [
            os.path.join(REPO_ROOT, "run/condor/mosei_joint_tf_kfold.args"),
            os.path.join(REPO_ROOT, "run/condor/mosei_joint_tf_nl8.args"),
        ],
    },
    {
        "dataset":  "MOSEI",
        "method":   "JointTF_IHA",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mosei/vt"),
        "args_files": [
            os.path.join(REPO_ROOT, "run/condor/mosei_joint_tf_iha_kfold.args"),
            os.path.join(REPO_ROOT, "run/condor/mosei_joint_tf_iha_nl8.args"),
        ],
    },
    {
        "dataset":  "MOSI",
        "method":   "JointTF",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mosi/vt"),
        "args_files": [
            os.path.join(REPO_ROOT, "run/condor/mosi_joint_tf_kfold.args"),
            os.path.join(REPO_ROOT, "run/condor/mosi_joint_tf_nl8.args"),
        ],
    },
    {
        "dataset":  "MOSI",
        "method":   "JointTF_IHA",
        "ckpt_dir": os.path.join(REPO_ROOT, "artifacts/models/factorcl/mosi/vt"),
        "args_files": [
            os.path.join(REPO_ROOT, "run/condor/mosi_joint_tf_iha_kfold.args"),
            os.path.join(REPO_ROOT, "run/condor/mosi_joint_tf_iha_nl8.args"),
        ],
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
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
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
    Parse .args file(s), check completeness, load metrics.
    Returns:
        fold_results: dict[fold_int -> dict[nlayers_int -> {"val": float, "test": float}]]
        missing: dict[nlayers_int -> list[int]]  (folds missing per nlayers)
    """
    ckpt_dir = exp["ckpt_dir"]
    args_files = exp.get("args_files", [exp.get("args_file")])

    # Collect all lines across all args files
    all_lines = []
    for args_file in args_files:
        if not os.path.exists(args_file):
            continue
        with open(args_file) as f:
            all_lines.extend([l.strip() for l in f if l.strip()])

    if not all_lines:
        print(f"  [ERROR] No args files found for {exp['dataset']} / {exp['method']}")
        return {}, {}, [], []

    # Parse each line → group by (fold, nlayers)
    model_prefix = None
    jobs = {}  # (fold, nlayers) -> ckpt_path

    for line in all_lines:
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

    all_nl_global = [1, 2, 4, 8]

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


def show_lrwd_grid_results():
    """
    Read the lrwd_grid scout checkpoints (6-7 combos × folds {0,2,4,6,8} × nlayers {1,2,4})
    and rank lr/wd choices by mean val accuracy per (dataset, method).
    Flags the currently-used lr/wd and prints a go/no-go recommendation.
    """
    SCOUT_FOLDS   = [0, 2, 4, 6, 8]
    SCOUT_NLAYERS = [1, 2, 4]
    SCOUT_GRID    = [
        ("0.0001", "0.0"),
        ("0.0001", "0.001"),
        ("0.0005", "0.0"),
        ("0.0005", "0.001"),
        ("0.001",  "0.0"),
        ("0.001",  "0.001"),
    ]
    # Currently-used lr/wd from the full kfold experiments
    CURRENT_LRWD = {
        ("Mustard",  "JointTF"):     ("0.0005",  "0.0"),
        ("Mustard",  "JointTF_IHA"): ("0.001",   "0.001"),
        ("UR-Funny", "JointTF"):     ("0.001",   "0.001"),
        ("UR-Funny", "JointTF_IHA"): ("0.001",   "0.001"),
        ("MOSEI",    "JointTF"):     ("0.001",   "0.0001"),
        ("MOSEI",    "JointTF_IHA"): ("0.0005",  "0.001"),
        ("MOSI",     "JointTF"):     ("0.001",   "0.001"),
        ("MOSI",     "JointTF_IHA"): ("0.0005",  "0.0"),
    }
    # Extra combos per experiment (added to SCOUT_GRID)
    EXTRA_GRID = {
        ("Mustard", "JointTF"): [("0.00005", "0.0")],
    }

    print("\n" + "=" * 80)
    print("LR/WD GRID SCOUT RESULTS (folds {0,2,4,6,8}, best-nlayer per fold)")
    print("=" * 80)

    datasets_order = ["Mustard", "UR-Funny", "MOSEI", "MOSI"]
    methods = ["JointTF", "JointTF_IHA"]

    for exp in EXPERIMENTS:
        dataset = exp["dataset"]
        method  = exp["method"]
        ckpt_dir = exp["ckpt_dir"]
        base_args = exp.get("args_files", [exp.get("args_file")])[0]
        args_file = base_args.replace("_kfold.args", "_lrwd_grid.args").replace("_kfold_v2.args", "_lrwd_grid.args")

        if not os.path.exists(args_file):
            continue  # scout not created yet

        grid = SCOUT_GRID + EXTRA_GRID.get((dataset, method), [])
        current = CURRENT_LRWD.get((dataset, method))

        print(f"\n{dataset} / {method}  (current lr/wd: lr={current[0]}, wd={current[1]})")

        # Read args file to get model prefix
        with open(args_file) as f:
            first_line = f.readline().strip()
        p0 = parse_args_line(first_line)
        model_prefix = model_name_from_config(p0["config"])

        # For each lr/wd combo, collect best-nlayer val_acc and test_acc per fold
        results_by_lrwd = {}  # (lr, wd) -> list of test_accs across folds
        val_by_lrwd     = {}  # (lr, wd) -> list of best val_accs across folds

        for lr, wd in grid:
            test_accs = []
            val_accs  = []
            for fold in SCOUT_FOLDS:
                best_val  = -1
                best_test = None
                for nl in SCOUT_NLAYERS:
                    # Reconstruct suffix manually for scout checkpoints
                    iha_suffix = ""
                    if "IHA" in method:
                        iha_suffix = "_phq4_phkv4_ihainitidentity_ihaLall"
                    fname = (f"{model_prefix}_fold{fold}_vldaccuracy"
                             f"_lr{lr}_wd{wd}_bs32{iha_suffix}_nlayers{nl}.pth.tar")
                    ckpt_path = os.path.join(ckpt_dir, fname)
                    if not os.path.exists(ckpt_path):
                        continue
                    v, t = extract_metrics(ckpt_path)
                    if v is not None and v > best_val:
                        best_val  = v
                        best_test = t
                if best_test is not None:
                    val_accs.append(best_val)
                    test_accs.append(best_test)

            results_by_lrwd[(lr, wd)] = test_accs
            val_by_lrwd[(lr, wd)]     = val_accs

        # Rank by mean val acc (only combos with data)
        ranked = sorted(
            [(lr, wd) for (lr, wd) in grid if val_by_lrwd.get((lr, wd))],
            key=lambda k: -np.mean(val_by_lrwd[k])
        )

        if not ranked:
            print("  No scout checkpoints found yet.")
            continue

        col_w = 14
        print(f"  {'lr':<8} {'wd':<8} {'n_folds':>7}  {'val mean':>9}  {'val std':>8}  "
              f"{'test mean':>10}  {'test std':>9}  {'flag'}")
        print("  " + "-" * 75)
        best_val_mean = np.mean(val_by_lrwd[ranked[0]])
        for rank_i, (lr, wd) in enumerate(ranked):
            v_accs = val_by_lrwd[(lr, wd)]
            t_accs = results_by_lrwd[(lr, wd)]
            vm = np.mean(v_accs) * 100
            vs = np.std(v_accs)  * 100
            tm = np.mean(t_accs) * 100 if t_accs else float("nan")
            ts = np.std(t_accs)  * 100 if t_accs else float("nan")
            flag = ""
            if (lr, wd) == current:
                flag += "[CURRENT]"
            if rank_i == 0:
                flag += " <-- BEST"
            print(f"  lr={lr:<7} wd={wd:<7} n={len(v_accs):>2}   "
                  f"{vm:>8.2f}%  {vs:>7.2f}%  {tm:>9.2f}%  {ts:>8.2f}%  {flag}")

        # Recommendation
        if len(ranked) >= 2:
            gap = (np.mean(val_by_lrwd[ranked[0]]) - np.mean(val_by_lrwd[ranked[1]])) * 100
            best_is_current = ranked[0] == current
            print(f"\n  Gap best vs 2nd: {gap:.2f}%")
            if best_is_current and gap > 1.0:
                print("  => KEEP current lr/wd (clearly best, gap > 1%)")
            elif best_is_current:
                print("  => KEEP current lr/wd (best, but gap is small — results are stable)")
            else:
                print(f"  => CONSIDER switching to lr={ranked[0][0]}, wd={ranked[0][1]} "
                      f"and running full 10-fold kfold for this experiment")


if __name__ == "__main__":
    main()
    show_lrwd_grid_results()
