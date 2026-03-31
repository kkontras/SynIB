#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy}"
if command -v conda >/dev/null 2>&1; then
  source ~/anaconda3/etc/profile.d/conda.sh || true
  conda activate "${CONDA_ENV_PATH}" || true
fi

PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/artifacts/reports}"
mkdir -p "${OUT_DIR}"
DATASETS_CSV="${DATASETS_CSV:-all}"

"${PYTHON_BIN}" - <<'PY'
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(os.getcwd())
OUT_DIR = Path(os.environ.get("OUT_DIR", REPO_ROOT / "artifacts" / "reports"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_CSV = os.environ.get("DATASETS_CSV", "all").strip().lower()

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FOLD_RE = re.compile(r"fold(?:fold)?(\d+)")
L_RE = re.compile(r"_l([0-9.]+)_vldaccuracy")
LR_RE = re.compile(r"_lr([0-9.]+)_wd")
WD_RE = re.compile(r"_wd([0-9.]+)(?:_|\.pth)")
LSPARSE_RE = re.compile(r"_lsparse([0-9.]+)")
PMIN_RE = re.compile(r"_pmin([0-9.]+)")
SAVE_DIR_RE = re.compile(r"save_dir:\s*([^\s]+)")
VAL_LINE_RE = re.compile(r"Val Epoch.*?Acc_combined:\s*([0-9]+(?:\.[0-9]+)?).*?(?:CEU_synergy|T_CEU_S):\s*([0-9]+(?:\.[0-9]+)?)")
TEST_LINE_RE = re.compile(r"Test Epoch.*?Acc_combined:\s*([0-9]+(?:\.[0-9]+)?).*?(?:CEU_synergy|T_CEU_S):\s*([0-9]+(?:\.[0-9]+)?)")

DATASETS = {
    "mosi": {
        "title": "MOSI",
        "checkpoint_root": Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/Balance_Final/MOSI/VT"),
        "u_log_dir": REPO_ROOT / "run/condor/logs/mosi_synibu",
        "u_nonpre_log_dir": REPO_ROOT / "run/condor/logs/mosi_synibu_nonpre",
    },
    "mosei": {
        "title": "MOSEI",
        "checkpoint_root": Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Rmask/MOSEI/VT"),
        "u_log_dir": REPO_ROOT / "run/condor/logs/mosei_synibu",
        "u_nonpre_log_dir": REPO_ROOT / "run/condor/logs/mosei_synibu_nonpre",
    },
    "urfunny": {
        "title": "URFunny",
        "checkpoint_root": Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/URFUNNY"),
        "u_log_dir": REPO_ROOT / "run/condor/logs/urfunny_synibu",
        "u_nonpre_log_dir": REPO_ROOT / "run/condor/logs/urfunny_synibu_nonpre",
    },
    "mustard": {
        "title": "MUStARD",
        "checkpoint_root": Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/MUSTARD"),
        "u_log_dir": REPO_ROOT / "run/condor/logs/mustard_synibu",
        "u_nonpre_log_dir": REPO_ROOT / "run/condor/logs/mustard_synibu_nonpre",
    },
}

VARIANT_ORDER = ["synib", "synib_nonpre", "synib_u", "synib_u_nonpre"]
MASKS = ["learned", "random"]


def expected_counts(variant, dataset_key, mask):
    if variant in {"synib_u", "synib_u_nonpre"}:
        return 36 if mask == "learned" else 18
    if variant in {"synib", "synib_nonpre"}:
        if variant == "synib_nonpre" and dataset_key in {"mustard", "urfunny"}:
            return 0
        return 28 if mask == "learned" else 20
    return 0


def to_float(value):
    if value is None:
        return math.nan
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return math.nan


def mean(values):
    vals = [float(v) for v in values if not math.isnan(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def detect_mask(name):
    lowered = name.lower()
    if "perturblearned" in lowered or "perturnlearned" in lowered:
        return "learned"
    if "perturbrandom" in lowered or "perturnrandom" in lowered:
        return "random"
    return "none"


def parse_name_fields(name):
    fold_match = FOLD_RE.search(name)
    return {
        "fold": int(fold_match.group(1)) if fold_match else None,
        "mask": detect_mask(name),
        "l": L_RE.search(name).group(1) if L_RE.search(name) else None,
        "lsparse": LSPARSE_RE.search(name).group(1) if LSPARSE_RE.search(name) else None,
        "pmin": PMIN_RE.search(name).group(1) if PMIN_RE.search(name) else None,
        "lr": LR_RE.search(name).group(1) if LR_RE.search(name) else None,
        "wd": WD_RE.search(name).group(1) if WD_RE.search(name) else None,
    }


def variant_from_checkpoint_name(name):
    if "SynIB_RMask_U" in name:
        return None
    if "SynIB_RMask_nonpre" in name or "SynIB_RMask_nopre" in name:
        return "synib_nonpre"
    if "SynIB_RMask" in name:
        return "synib"
    return None


def metrics_from_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    logs = ckpt.get("logs", {})
    best = logs.get("best_logs", {}).get("best_vaccuracy", {})
    post = ckpt.get("post_test_results", {})
    val_acc = to_float(best.get("acc", {}).get("combined")) * 100.0
    test_acc = to_float(post.get("acc", {}).get("combined")) * 100.0
    test_ceu = to_float(post.get("ceu", {}).get("combined", {}).get("synergy"))
    if math.isnan(test_ceu):
        step = best.get("step")
        test_logs = logs.get("test_logs", {})
        if step in test_logs:
            test_ceu = to_float(test_logs[step].get("ceu", {}).get("combined", {}).get("synergy"))
            if math.isnan(test_acc):
                test_acc = to_float(test_logs[step].get("acc", {}).get("combined")) * 100.0
    return {
        "mean_val_acc": val_acc,
        "mean_test_acc": test_acc,
        "mean_test_ceu_synergy": test_ceu,
    }


def collect_checkpoint_runs(dataset_key, root):
    by_variant = defaultdict(dict)
    if not root.exists():
        return by_variant
    for path in root.glob("SynIB_RMask*.pth.tar"):
        variant = variant_from_checkpoint_name(path.name)
        if variant is None:
            continue
        fields = parse_name_fields(path.name)
        if fields["fold"] is None:
            continue
        try:
            metrics = metrics_from_checkpoint(path)
        except Exception:
            continue
        record = {
            "dataset": dataset_key,
            "variant": variant,
            **fields,
            **metrics,
            "source": str(path),
            "mtime": path.stat().st_mtime,
        }
        key = (fields["mask"], fields["l"], fields["lsparse"], fields["pmin"], fields["lr"], fields["wd"], fields["fold"])
        prev = by_variant[variant].get(key)
        if prev is None or record["mtime"] > prev["mtime"]:
            by_variant[variant][key] = record
    return by_variant


def parse_finished_u_log(path, variant):
    text = path.read_text(errors="ignore")
    clean = ANSI_RE.sub("", text)
    if "Job finished" not in clean and "We are in the final state." not in clean:
        return None
    save_dirs = SAVE_DIR_RE.findall(clean)
    if not save_dirs:
        return None
    val_matches = VAL_LINE_RE.findall(clean)
    test_matches = TEST_LINE_RE.findall(clean)
    if not val_matches or not test_matches:
        return None
    save_dir = save_dirs[-1]
    fields = parse_name_fields(save_dir)
    if fields["fold"] is None:
        return None
    val_acc, _ = val_matches[-1]
    test_acc, test_ceu = test_matches[-1]
    return {
        "variant": variant,
        **fields,
        "mean_val_acc": float(val_acc),
        "mean_test_acc": float(test_acc),
        "mean_test_ceu_synergy": float(test_ceu),
        "source": str(path),
        "mtime": path.stat().st_mtime,
    }


def collect_u_runs(log_dir, variant, dataset_key):
    records = {}
    if not log_dir.exists():
        return records
    for path in log_dir.glob("*.out"):
        parsed = parse_finished_u_log(path, variant)
        if parsed is None:
            continue
        parsed["dataset"] = dataset_key
        key = (parsed["mask"], parsed["l"], parsed["lsparse"], parsed["pmin"], parsed["lr"], parsed["wd"], parsed["fold"])
        prev = records.get(key)
        if prev is None or parsed["mtime"] > prev["mtime"]:
            records[key] = parsed
    return records


def aggregate_variant(records):
    grouped = defaultdict(list)
    for rec in records.values():
        cfg_key = (rec["mask"], rec["l"], rec["lsparse"], rec["pmin"], rec["lr"], rec["wd"])
        grouped[cfg_key].append(rec)
    out = []
    for cfg_key, items in grouped.items():
        folds = sorted({item["fold"] for item in items if item["fold"] is not None})
        out.append({
            "mask": cfg_key[0],
            "l": cfg_key[1],
            "lsparse": cfg_key[2],
            "pmin": cfg_key[3],
            "lr": cfg_key[4],
            "wd": cfg_key[5],
            "folds": folds,
            "n_folds": len(folds),
            "is_complete": len(folds) == 3,
            "mean_val_acc": mean([item["mean_val_acc"] for item in items]),
            "mean_test_acc": mean([item["mean_test_acc"] for item in items]),
            "mean_test_ceu_synergy": mean([item["mean_test_ceu_synergy"] for item in items]),
        })
    return out


def best_config(rows, metric):
    valid = [row for row in rows if row["is_complete"] and not math.isnan(row[metric])]
    if not valid:
        return None
    return max(
        valid,
        key=lambda row: (
            row[metric],
            row["n_folds"],
            row["mean_test_acc"] if not math.isnan(row["mean_test_acc"]) else -1e9,
            row["mean_val_acc"] if not math.isnan(row["mean_val_acc"]) else -1e9,
        ),
    )


def summarize_mask(rows, variant, dataset_key, mask):
    mask_rows = [row for row in rows if row["mask"] == mask]
    complete_rows = [row for row in mask_rows if row["is_complete"]]
    expected = expected_counts(variant, dataset_key, mask)
    complete_count = len(complete_rows)
    incomplete_count = len([row for row in mask_rows if not row["is_complete"]])
    missing_count = max(expected - complete_count - incomplete_count, 0)
    best = best_config(complete_rows, "mean_val_acc")
    return {
        "expected_hp_runs": expected,
        "complete_hp_runs": complete_count,
        "incomplete_hp_runs": incomplete_count,
        "missing_hp_runs": missing_count,
        "best": best,
    }


def summarize_vanilla(rows):
    vanilla = [row for row in rows if row["l"] == "0" and row["is_complete"]]
    if not vanilla:
        return None
    return max(
        vanilla,
        key=lambda row: (
            row["mean_val_acc"] if not math.isnan(row["mean_val_acc"]) else -1e9,
            row["mean_test_acc"] if not math.isnan(row["mean_test_acc"]) else -1e9,
        ),
    )


def format_hparams(row):
    parts = []
    if row.get("lr"):
        parts.append(f"lr={row['lr']}")
    if row.get("wd"):
        parts.append(f"wd={row['wd']}")
    if row.get("l"):
        parts.append(f"l={row['l']}")
    if row.get("lsparse"):
        parts.append(f"lsparse={row['lsparse']}")
    if row.get("pmin"):
        parts.append(f"pmin={row['pmin']}")
    return ", ".join(parts) if parts else "-"


def fmt_num(value, digits=2):
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


all_rows = []
markdown_parts = []
text_parts = []


def build_fixed_table(rows, headers, keys):
    str_rows = []
    widths = [len(h) for h in headers]
    for row in rows:
        vals = [str(row.get(k, "")) for k in keys]
        str_rows.append(vals)
        widths = [max(w, len(v)) for w, v in zip(widths, vals)]

    def fmt_line(vals):
        return " | ".join(v.ljust(w) for v, w in zip(vals, widths))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_line(headers), sep]
    out.extend(fmt_line(vals) for vals in str_rows)
    return "\n".join(out)

for dataset_key, meta in DATASETS.items():
    if DATASETS_CSV != "all":
        selected = {part.strip() for part in DATASETS_CSV.split(",") if part.strip()}
        if dataset_key not in selected:
            continue
    variant_runs = defaultdict(dict)
    ckpt_runs = collect_checkpoint_runs(dataset_key, meta["checkpoint_root"])
    for variant, records in ckpt_runs.items():
        variant_runs[variant].update(records)
    variant_runs["synib_u"].update(collect_u_runs(meta["u_log_dir"], "synib_u", dataset_key))
    variant_runs["synib_u_nonpre"].update(collect_u_runs(meta["u_nonpre_log_dir"], "synib_u_nonpre", dataset_key))

    markdown_parts.append(f"## {meta['title']}\n")
    vanilla_rows = []
    dataset_rows = []

    for variant in VARIANT_ORDER:
        rows = aggregate_variant(variant_runs.get(variant, {}))
        vanilla = summarize_vanilla(rows)
        if vanilla is not None:
            vanilla_rows.append({
                "variant": variant,
                "hyperparams": format_hparams(vanilla),
                "folds": ",".join(str(x) for x in vanilla["folds"]),
                "val_acc": fmt_num(vanilla["mean_val_acc"]),
                "test_acc": fmt_num(vanilla["mean_test_acc"]),
                "test_ceu": fmt_num(vanilla["mean_test_ceu_synergy"], 4),
            })
        for mask in MASKS:
            summary = summarize_mask(rows, variant, dataset_key, mask)
            best = summary["best"]
            if best is None:
                row = {
                    "dataset": dataset_key,
                    "variant": variant,
                    "mask": mask,
                    "expected_hp_runs": summary["expected_hp_runs"],
                    "complete_hp_runs": summary["complete_hp_runs"],
                    "incomplete_hp_runs": summary["incomplete_hp_runs"],
                    "missing_hp_runs": summary["missing_hp_runs"],
                    "hyperparams": "-",
                    "folds_used": "-" if summary["incomplete_hp_runs"] == 0 else "<3 folds only",
                    "mean_val_acc": math.nan,
                    "mean_test_acc": math.nan,
                    "mean_test_ceu_synergy": math.nan,
                }
            else:
                row = {
                    "dataset": dataset_key,
                    "variant": variant,
                    "mask": mask,
                    "expected_hp_runs": summary["expected_hp_runs"],
                    "complete_hp_runs": summary["complete_hp_runs"],
                    "incomplete_hp_runs": summary["incomplete_hp_runs"],
                    "missing_hp_runs": summary["missing_hp_runs"],
                    "hyperparams": format_hparams(best),
                    "folds_used": ",".join(str(x) for x in best["folds"]),
                    "mean_val_acc": best["mean_val_acc"],
                    "mean_test_acc": best["mean_test_acc"],
                    "mean_test_ceu_synergy": best["mean_test_ceu_synergy"],
                }
            all_rows.append(row)
            dataset_rows.append({
                "variant": row["variant"],
                "mask": row["mask"],
                "expected": row["expected_hp_runs"],
                "complete": row["complete_hp_runs"],
                "incomplete": row["incomplete_hp_runs"],
                "missing": row["missing_hp_runs"],
                "best_val_hparams": row["hyperparams"],
                "folds": row["folds_used"],
                "val_acc": fmt_num(row["mean_val_acc"]),
                "test_acc": fmt_num(row["mean_test_acc"]),
                "test_ceu": fmt_num(row["mean_test_ceu_synergy"], 4),
            })

    headers = [
        "variant", "mask", "expected", "complete", "incomplete", "missing",
        "best_val_hparams", "folds", "val_acc", "test_acc", "test_ceu"
    ]
    keys = [
        "variant", "mask", "expected", "complete", "incomplete", "missing",
        "best_val_hparams", "folds", "val_acc", "test_acc", "test_ceu"
    ]
    if vanilla_rows:
        vanilla_headers = ["variant", "vanilla_hparams", "folds", "val_acc", "test_acc", "test_ceu"]
        vanilla_keys = ["variant", "hyperparams", "folds", "val_acc", "test_acc", "test_ceu"]
        vanilla_table = build_fixed_table(vanilla_rows, vanilla_headers, vanilla_keys)
        markdown_parts.append("Vanilla `--l 0`\n")
        markdown_parts.append("```text")
        markdown_parts.append(vanilla_table)
        markdown_parts.append("```")
        markdown_parts.append("")
        text_parts.append(f"{meta['title']} - vanilla")
        text_parts.append(vanilla_table)
        text_parts.append("")

    fixed_table = build_fixed_table(dataset_rows, headers, keys)
    markdown_parts.append("```text")
    markdown_parts.append(fixed_table)
    markdown_parts.append("```")
    markdown_parts.append("")
    text_parts.append(meta["title"])
    text_parts.append(fixed_table)
    text_parts.append("")

md_path = OUT_DIR / "factorcl_synib_family_tables.md"
tsv_path = OUT_DIR / "factorcl_synib_family_rows.tsv"
txt_path = OUT_DIR / "factorcl_synib_family_tables.txt"

md_text = "\n".join(markdown_parts).rstrip() + "\n"
md_path.write_text(md_text)
txt_text = "\n".join(text_parts).rstrip() + "\n"
txt_path.write_text(txt_text)

with tsv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "variant",
            "mask",
            "expected_hp_runs",
            "complete_hp_runs",
            "incomplete_hp_runs",
            "missing_hp_runs",
            "hyperparams",
            "folds_used",
            "mean_val_acc",
            "mean_test_acc",
            "mean_test_ceu_synergy",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(all_rows)

print(md_text)
print(f"Wrote {md_path}")
print(f"Wrote {txt_path}")
print(f"Wrote {tsv_path}")
PY
