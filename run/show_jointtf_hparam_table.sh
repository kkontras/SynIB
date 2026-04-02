#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASETS_CSV="${DATASETS_CSV:-all}"

python3 - <<'PY'
import os
import re
import math
from pathlib import Path
import torch

REPO_ROOT = Path(os.getcwd())
DATASETS_CSV = os.environ.get("DATASETS_CSV", "all").strip().lower()

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

DATASETS = {
    "mustard": ("MUStARD", REPO_ROOT / "run/condor/logs/mustard_joint_tf",  REPO_ROOT / "run/condor/logs/mustard_joint_tf_iha",  None,                                                                      None),
    "mosi":    ("MOSI",    REPO_ROOT / "run/condor/logs/mosi_joint_tf",     REPO_ROOT / "run/condor/logs/mosi_joint_tf_iha",     None,                                                                      None),
    "mosei":   ("MOSEI",   REPO_ROOT / "run/condor/logs/mosei_joint_tf",    REPO_ROOT / "run/condor/logs/mosei_joint_tf_iha",    None,                                                                      None),
    "urfunny": ("URFunny", REPO_ROOT / "run/condor/logs/urfunny_joint_tf",  REPO_ROOT / "run/condor/logs/urfunny_joint_tf_iha",  REPO_ROOT / "run/condor/logs/urfunny_joint_tf_iha_hsearch",  REPO_ROOT / "run/condor/urfunny_joint_tf_iha_hsearch.args"),
}

SAVE_DIR_RE   = re.compile(r"save_dir:\s*(\S+)")
FOLD_RE       = re.compile(r"_fold(\d+)_")
LR_RE         = re.compile(r"_lr([0-9.e+-]+)_wd")
WD_RE         = re.compile(r"_wd([0-9.e+-]+)_bs")
PHQ_RE        = re.compile(r"_phq(\d+)_")
PHKV_RE       = re.compile(r"_phkv(\d+)_")
IHAINIT_RE    = re.compile(r"_ihainit(identity_noise|identity|orthogonal)_")
IHALR_RE      = re.compile(r"_ihalr([0-9.e+-]+)_")
IHAL_RE       = re.compile(r"_ihaL([^_.]+)(?:\.pth|_)")

FINAL_VAL_RE   = re.compile(r"Val Epoch.*?step 0.*?Acc_combined:\s*([0-9]+(?:\.[0-9]+)?)")
FINAL_TEST_RE  = re.compile(r"Test Epoch.*?step 0.*?Acc_combined:\s*([0-9]+(?:\.[0-9]+)?)")
MODEL_PATH_RE  = re.compile(r"Model saved successfully at:\s*(\S+)")


def train_acc_from_checkpoint(model_path_str):
    """Load .pth.tar and return training accuracy at the best validation step (0-100 scale)."""
    p = Path(model_path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return math.nan
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        logs = ckpt.get("logs", {})
        best_step = logs.get("best_logs", {}).get("best_vaccuracy", {}).get("step")
        if best_step is None:
            return math.nan
        train_acc = logs.get("train_logs", {}).get(best_step, {}).get("acc", {}).get("combined")
        if train_acc is None:
            return math.nan
        return float(train_acc) * 100
    except Exception:
        return math.nan


def parse_log(path, iha_fields=False):
    text = ANSI_RE.sub("", path.read_text(errors="ignore"))
    if "Job finished" not in text and "We are in the final state." not in text:
        return None
    m = SAVE_DIR_RE.search(text)
    if not m:
        return None
    sd = m.group(1)
    fold_m = FOLD_RE.search(sd)
    lr_m   = LR_RE.search(sd)
    wd_m   = WD_RE.search(sd)
    if not (fold_m and lr_m and wd_m):
        return None
    val_matches  = FINAL_VAL_RE.findall(text)
    test_matches = FINAL_TEST_RE.findall(text)
    if not val_matches or not test_matches:
        return None
    model_m = MODEL_PATH_RE.search(text)
    trn = train_acc_from_checkpoint(model_m.group(1)) if model_m else math.nan
    rec = {
        "fold":  int(fold_m.group(1)),
        "lr":    lr_m.group(1),
        "wd":    wd_m.group(1),
        "val":   float(val_matches[-1]),
        "test":  float(test_matches[-1]),
        "trn":   trn,
        "mtime": path.stat().st_mtime,
    }
    if iha_fields:
        rec["phq"]    = PHQ_RE.search(sd).group(1)    if PHQ_RE.search(sd)     else "-"
        rec["phkv"]   = PHKV_RE.search(sd).group(1)   if PHKV_RE.search(sd)    else "-"
        rec["ihainit"]= IHAINIT_RE.search(sd).group(1) if IHAINIT_RE.search(sd) else "-"
        rec["ihalr"]  = IHALR_RE.search(sd).group(1)  if IHALR_RE.search(sd)   else "-"
        rec["ihaL"]   = IHAL_RE.search(sd).group(1)   if IHAL_RE.search(sd)    else "-"
    return rec


def collect_runs(log_dir, iha_fields=False):
    records = {}
    if not log_dir or not log_dir.exists():
        return records
    for path in log_dir.glob("*.out"):
        rec = parse_log(path, iha_fields=iha_fields)
        if rec is None:
            continue
        if iha_fields:
            key = (rec["lr"], rec["wd"], rec["phq"], rec["phkv"],
                   rec["ihainit"], rec["ihalr"], rec["ihaL"], rec["fold"])
        else:
            key = (rec["lr"], rec["wd"], rec["fold"])
        if key not in records or rec["mtime"] > records[key]["mtime"]:
            records[key] = rec
    return records


def fmt(v):
    return f"{v:6.2f}" if not math.isnan(v) else "  -   "


def fmt_mean_std(vals):
    valid = [x for x in vals if not math.isnan(x)]
    if not valid:
        return "   -       "
    mean = sum(valid) / len(valid)
    if len(valid) > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in valid) / (len(valid) - 1))
        return f"{mean:5.2f}±{std:4.2f}"
    return f"{mean:5.2f}±  - "


def nan():
    return math.nan


def print_method_table(method_name, records, n_folds=3):
    combos = sorted({(r["lr"], r["wd"]) for r in records.values()},
                    key=lambda x: (float(x[0]), float(x[1])))
    if not combos:
        print(f"\n  [{method_name}]  (no finished runs)")
        return None

    folds = list(range(n_folds))
    fold_headers = "  ".join(f"f{f}_trn  f{f}_val  f{f}_tst" for f in folds)
    header = f"  {'lr':>8}  {'wd':>8}  |  {fold_headers}  |  avg_trn        avg_val        avg_tst"
    print(f"\n  [{method_name}]")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    best = None
    for lr, wd in combos:
        vals, tests, trns, parts = [], [], [], []
        for f in folds:
            rec = records.get((lr, wd, f))
            v = rec["val"] if rec else nan()
            t = rec["test"] if rec else nan()
            r = rec["trn"] if rec else nan()
            vals.append(v); tests.append(t); trns.append(r)
            parts.append(f"{fmt(r)}  {fmt(v)}  {fmt(t)}")
        n_done = sum(1 for x in vals if not math.isnan(x))
        avg_v  = sum(x for x in vals  if not math.isnan(x)) / max(n_done, 1)
        print(f"  {lr:>10}  {wd:>8}  |  {'  '.join(parts)}  |  {fmt_mean_std(trns)}  {fmt_mean_std(vals)}  {fmt_mean_std(tests)}  ({n_done}/{len(folds)})")
        if n_done == len(folds) and (best is None or avg_v > best["avg_val"]):
            best = {"lr": lr, "wd": wd, "avg_val": avg_v, "avg_test": sum(x for x in tests if not math.isnan(x)) / max(n_done, 1), "vals": vals, "tests": tests, "trns": trns}
    return best


def parse_expected_hsearch_combos(args_file):
    """Return sorted list of unique (lr, wd, phq, phkv, ihainit, ihalr, ihaL) from an args file."""
    if args_file is None or not args_file.exists():
        return []
    combos = set()
    for line in args_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # reuse same regexes on the raw arg line (field names match save_dir patterns)
        lr_m     = re.search(r"--lr\s+([0-9.e+-]+)", line)
        wd_m     = re.search(r"--wd\s+([0-9.e+-]+)", line)
        phq_m    = re.search(r"--pseudo_heads_q\s+(\d+)", line)
        phkv_m   = re.search(r"--pseudo_heads_kv\s+(\d+)", line)
        init_m   = re.search(r"--iha_init\s+(\S+)", line)
        ihalr_m  = re.search(r"--iha_lr\s+([0-9.e+-]+)", line)
        ihaL_m   = re.search(r"--iha_layers\s+(\S+)", line)
        if not (lr_m and wd_m and phq_m and phkv_m and init_m and ihaL_m):
            continue
        # normalise layers: "2,3" → "2-3" to match save_dir encoding
        ihaL = ihaL_m.group(1).replace(",", "-")
        combos.add((
            lr_m.group(1),
            wd_m.group(1),
            phq_m.group(1),
            phkv_m.group(1),
            init_m.group(1),
            ihalr_m.group(1) if ihalr_m else "-",
            ihaL,
        ))
    return sorted(combos, key=lambda x: (float(x[0]), float(x[1]), int(x[2]), int(x[3]), x[4], x[5], x[6]))


def print_iha_hsearch_table(method_name, records, n_folds=3, args_file=None):
    # Key: (lr, wd, phq, phkv, ihainit, ihalr, ihaL)
    combos_from_args = set(parse_expected_hsearch_combos(args_file))
    combos_from_runs = {(r["lr"], r["wd"], r["phq"], r["phkv"], r["ihainit"], r["ihalr"], r["ihaL"])
                        for r in records.values()}
    # Args file is source of truth; fall back to runs if no args file provided
    all_combos = combos_from_args if combos_from_args else combos_from_runs
    combos = sorted(all_combos, key=lambda x: (float(x[0]), float(x[1]), int(x[2]), int(x[3]), x[4], x[5], x[6]))
    if not combos:
        print(f"\n  [{method_name}]  (no finished runs)")
        return None

    folds = list(range(n_folds))
    fold_headers = "  ".join(f"f{f}_trn  f{f}_val  f{f}_tst" for f in folds)
    header = (f"  {'lr':>8}  {'wd':>8}  {'phq':>4}  {'phkv':>4}  {'ihainit':<16}"
              f"  {'ihalr':>8}  {'ihaL':>5}  |  {fold_headers}  |  avg_trn        avg_val        avg_tst")
    print(f"\n  [{method_name}]")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    best = None
    for lr, wd, phq, phkv, ihainit, ihalr, ihaL in combos:
        vals, tests, trns, parts = [], [], [], []
        for f in folds:
            rec = records.get((lr, wd, phq, phkv, ihainit, ihalr, ihaL, f))
            v = rec["val"] if rec else nan()
            t = rec["test"] if rec else nan()
            r = rec["trn"] if rec else nan()
            vals.append(v); tests.append(t); trns.append(r)
            parts.append(f"{fmt(r)}  {fmt(v)}  {fmt(t)}")
        n_done = sum(1 for x in vals if not math.isnan(x))
        avg_v  = sum(x for x in vals  if not math.isnan(x)) / max(n_done, 1)
        print(f"  {lr:>10}  {wd:>8}  {phq:>4}  {phkv:>4}  {ihainit:<16}"
              f"  {ihalr:>8}  {ihaL:>5}  |  {'  '.join(parts)}  |"
              f"  {fmt_mean_std(trns)}  {fmt_mean_std(vals)}  {fmt_mean_std(tests)}  ({n_done}/{len(folds)})")
        if n_done == len(folds) and (best is None or avg_v > best["avg_val"]):
            best = {"lr": lr, "wd": wd, "phq": phq, "phkv": phkv,
                    "ihainit": ihainit, "ihalr": ihalr, "ihaL": ihaL,
                    "avg_val": avg_v, "avg_test": sum(x for x in tests if not math.isnan(x)) / max(n_done, 1),
                    "vals": vals, "tests": tests, "trns": trns}
    return best


selected = set()
if DATASETS_CSV != "all":
    selected = {s.strip() for s in DATASETS_CSV.split(",") if s.strip()}

summary = []

for ds_key, (title, tf_dir, iha_dir, hsearch_dir, hsearch_args) in DATASETS.items():
    if selected and ds_key not in selected:
        continue
    print(f"\n{'='*70}")
    print(f"  Dataset: {title}")
    print(f"{'='*70}")

    tf_records     = collect_runs(tf_dir)
    iha_records    = collect_runs(iha_dir)
    hsearch_records = collect_runs(hsearch_dir, iha_fields=True) if hsearch_dir else {}

    if not tf_records and not iha_records and not hsearch_records:
        print("  (no finished runs found)")
        continue

    best_tf  = print_method_table("joint_tf",     tf_records)
    best_iha = print_method_table("joint_tf_iha", iha_records)
    summary.append((title, "joint_tf",     best_tf,  None))
    summary.append((title, "joint_tf_iha", best_iha, None))

    if hsearch_records or (hsearch_args and hsearch_args.exists()):
        best_hs = print_iha_hsearch_table("joint_tf_iha_hsearch", hsearch_records, args_file=hsearch_args)
        summary.append((title, "joint_tf_iha_hsearch", best_hs, best_hs))

# ── Best-per-method summary ──────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("  BEST VALIDATION SUMMARY  (complete folds only, avg over folds)")
print(f"{'='*70}")
header = f"  {'dataset':<12}  {'method':<22}  {'lr':>8}  {'wd':>8}  |  avg_trn        avg_val        avg_tst        | per-fold val"
print(header)
print("  " + "-" * (len(header) - 2))
for title, method, best, best_full in summary:
    if best is None:
        print(f"  {title:<12}  {method:<22}  {'':>8}  {'':>8}  |    -              -              -")
        continue
    fold_vals = "  ".join(fmt(v) for v in best["vals"])
    extra = ""
    if best_full is not None:
        extra = (f"  phq={best_full['phq']} phkv={best_full['phkv']}"
                 f" init={best_full['ihainit']} ihalr={best_full['ihalr']} L={best_full['ihaL']}")
    trns = best.get("trns", [math.nan] * len(best["vals"]))
    print(f"  {title:<12}  {method:<22}  {best['lr']:>8}  {best['wd']:>8}  |"
          f"  {fmt_mean_std(trns)}  {fmt_mean_std(best['vals'])}  {fmt_mean_std(best['tests'])}  |  {fold_vals}{extra}")
print()
PY
