#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="${1:-/scratch/kkontras/checkpoints/synergy/MUSTARD}"

if [ ! -d "$CKPT_DIR" ]; then
  echo "Checkpoint directory not found: $CKPT_DIR" >&2
  exit 1
fi

cd "$CKPT_DIR"

python - <<'PY'
import os
import torch
from collections import defaultdict


def group_name(f):
    if "Cache_IHA_LoRA" in f:
        return "IHA_LoRA"
    if "Cache_IHA_" in f:
        return "IHA"
    if "Cache_SynIB_LoRa" in f:
        return "SynIB_LoRa"
    if "Cache_Text_LoRa" in f:
        return "Text_LoRa"
    if "Cache_Video_LoRa" in f:
        return "Video_LoRa"
    if "Cache_Zero" in f:
        return "Zero"
    if "Cache_LoRa" in f:
        return "LoRa"
    if "combined_lora" in f:
        return "combined_lora"
    if "synib_learned" in f:
        return "synib_learned"
    return "Other"


def extract_val_payload(ckpt):
    logs = ckpt.get("logs", {})
    best_logs = logs.get("best_logs", {})
    if "best_vaccuracy" in best_logs:
        return best_logs["best_vaccuracy"]
    return best_logs


def extract_combined_acc(payload):
    acc = payload.get("acc")
    if isinstance(acc, dict) and acc:
        for key in ("combined", "total"):
            value = acc.get(key)
            if isinstance(value, (int, float)):
                return float(value), f"acc.{key}"
        numeric = [(k, v) for k, v in acc.items() if isinstance(v, (int, float))]
        if numeric:
            k, v = max(numeric, key=lambda kv: kv[1])
            return float(v), f"acc.{k}"
    for k in ("accuracy", "acc", "best_accuracy", "best_acc"):
        v = payload.get(k)
        if isinstance(v, (int, float)):
            return float(v), k
    return None, None


def extract_test_payload(ckpt, val_payload):
    logs = ckpt.get("logs", {})
    test_logs = logs.get("test_logs", {})
    if not isinstance(test_logs, dict):
        return {}

    step = val_payload.get("step")
    if step in test_logs:
        payload = test_logs[step]
    elif str(step) in test_logs:
        payload = test_logs[str(step)]
    else:
        return {}

    if "acc" not in payload and "test_acc" in payload:
        payload = {k.replace("test_", ""): v for k, v in payload.items()}
    return payload


def fmt_score(score):
    return "NA" if score is None else f"{score:.6f}"


best = {}
all_rows = defaultdict(list)
files = sorted(f for f in os.listdir(".") if f.endswith(".pth.tar"))
total = len(files)

print(f"Scanning {total} checkpoint files in {os.getcwd()}", flush=True)

for idx, f in enumerate(files, start=1):
    print(f"[{idx}/{total}] opening {f}", flush=True)
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[{idx}/{total}] failed {f}: {e}", flush=True)
        all_rows["FAILED"].append((None, f, str(e)))
        continue

    group = group_name(f)
    val_payload = extract_val_payload(ckpt)
    test_payload = extract_test_payload(ckpt, val_payload)
    val_score, val_key = extract_combined_acc(val_payload)
    test_score, test_key = extract_combined_acc(test_payload)
    print(
        f"[{idx}/{total}] done group={group} "
        f"val_acc_combined={fmt_score(val_score)} "
        f"test_acc_combined={fmt_score(test_score)}",
        flush=True,
    )
    all_rows[group].append((val_score, test_score, f, val_key, test_key))

    if val_score is None:
        continue
    if group not in best or val_score > best[group][0]:
        best[group] = (val_score, test_score, f, val_key, test_key)

print("BEST PER CATEGORY")
for group in sorted(best):
    val_score, test_score, f, val_key, test_key = best[group]
    print(
        f"{group}\tval_acc_combined={fmt_score(val_score)}\t"
        f"test_acc_combined={fmt_score(test_score)}\t{f}"
    )

print("\nTOP RUNS PER CATEGORY")
for group in sorted(k for k in all_rows if k != "FAILED"):
    print(f"\n[{group}]")
    rows = sorted(all_rows[group], key=lambda x: (-1 if x[0] is None else -x[0], x[2]))
    for val_score, test_score, f, val_key, test_key in rows[:10]:
        print(
            f"val_acc_combined={fmt_score(val_score)}\t"
            f"test_acc_combined={fmt_score(test_score)}\t{f}"
        )

if "FAILED" in all_rows:
    print("\n[FAILED]")
    for _, f, err in all_rows["FAILED"]:
        print(f"{f}\t{err}")
PY
