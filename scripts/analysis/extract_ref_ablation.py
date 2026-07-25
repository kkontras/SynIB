#!/usr/bin/env python3
"""Extract per-epoch curves + final summary for the reference-distribution ablation.

Scans checkpoints whose filename contains a REFABL tag, reads ckpt["logs"]
(val_logs/test_logs per validation step), and writes:
  - docs/rebuttal_ref_ablation_curves.csv   (one row per run x val-step)
  - docs/rebuttal_ref_ablation_summary.csv  (one row per run: test metrics at best-val step)

Usage: PYTHONPATH=src python scripts/analysis/extract_ref_ablation.py [--glob PATTERN]
"""

import argparse
import csv
import glob
import os
import re

import torch

ROOTS = [
    "./artifacts/models/multibench/ur_funny/vt",
    "./artifacts/models/multibench/mustard/vt",
    "./artifacts/models/multibench/mosi/vt",
    "./data/data/2025_data/synergy/HatefulMemes",
    "./data/data/2025_data/synergy/CremadPlus/v2",
]

NAME_RE = re.compile(
    r"REFABL_(?P<ds>[a-z_0-9.]+?)(?:_lam(?P<lam>[0-9.]+)x)?_fold(?P<fold>\d)"
    r".*?_ref(?P<ref>uniform|class_prior|unimodal_anchor|anchor_legacy)")

# CREMA-D-Irony: irony class id = 6 (7-way head, irony appended last)
IRONY_CLASS = 6


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def row_from_metrics(m, ds):
    r = {
        "acc": g(m, "acc", "combined"),
        "f1": g(m, "f1", "combined"),
        "syn_subset_acc": g(m, "pg_acc", "combined", "group_metrics", "synergy", "internal_acc"),
        "syn_subset_n_contrib": g(m, "pg_acc", "combined", "group_metrics", "synergy", "contribution_to_total"),
        "diag_kl_1": g(m, "loss", "diag_kl_1"),
        "diag_kl_2": g(m, "loss", "diag_kl_2"),
        "kl_rand_1": g(m, "loss", "kl_synergy_rand_1"),
        "kl_rand_2": g(m, "loss", "kl_synergy_rand_2"),
        "ce_combined": g(m, "loss", "ce_loss_combined"),
    }
    fpc = g(m, "f1_perclass", "combined")
    if fpc is not None and hasattr(fpc, "numel") and fpc.numel() > IRONY_CLASS:
        r["irony_f1"] = float(fpc[IRONY_CLASS])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*REFABL*")
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args()

    curves, summary = [], []
    files = sorted(f for root in ROOTS for f in glob.glob(os.path.join(root, args.glob)))
    print(f"found {len(files)} checkpoints")
    for f in files:
        mname = NAME_RE.search(os.path.basename(f))
        meta = mname.groupdict() if mname else {"ds": "?", "fold": "?", "ref": "?", "lam": None}
        # ironic_rate from filename if present
        ir = re.search(r"_ir([0-9.]+)", os.path.basename(f))
        meta["alpha"] = ir.group(1) if ir else ""
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"SKIP {f}: {e}")
            continue
        logs = ck.get("logs", {})
        vl, tl = logs.get("val_logs", {}), logs.get("test_logs", {})
        best_step, best_vacc = None, -1.0
        for step in sorted(vl):
            vm, tm = vl[step], tl.get(step, {})
            base = {"file": os.path.basename(f), "step": step, **meta}
            vrow = {f"val_{k}": v for k, v in row_from_metrics(vm, meta["ds"]).items()}
            trow = {f"test_{k}": v for k, v in row_from_metrics(tm, meta["ds"]).items()}
            curves.append({**base, **vrow, **trow})
            vacc = g(vm, "acc", "combined") or -1
            if vacc > best_vacc:
                best_vacc, best_step = vacc, step
        if best_step is not None:
            tm = tl.get(best_step, {})
            summary.append({
                "file": os.path.basename(f), **meta,
                "best_val_step": best_step, "best_val_acc": best_vacc,
                **{f"test_{k}": v for k, v in row_from_metrics(tm, meta["ds"]).items()},
                "seed": logs.get("seed"),
            })

    os.makedirs(args.outdir, exist_ok=True)
    for name, rows in (("rebuttal_ref_ablation_curves.csv", curves),
                       ("rebuttal_ref_ablation_summary.csv", summary)):
        if not rows:
            print(f"{name}: no rows")
            continue
        cols = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("file", "ds", "ref", "fold", "alpha", "lam", "step"), c))
        p = os.path.join(args.outdir, name)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
