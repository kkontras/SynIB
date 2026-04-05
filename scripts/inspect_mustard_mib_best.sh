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


def extract_score(val_payload):
    acc = val_payload.get("acc")
    if isinstance(acc, dict) and acc:
        if "total" in acc and isinstance(acc["total"], (int, float)):
            return float(acc["total"]), "acc.total"
        numeric = [(k, v) for k, v in acc.items() if isinstance(v, (int, float))]
        if numeric:
            k, v = max(numeric, key=lambda kv: kv[1])
            return float(v), f"acc.{k}"
    for k in ("accuracy", "acc", "best_accuracy", "best_acc"):
        v = val_payload.get(k)
        if isinstance(v, (int, float)):
            return float(v), k
    return None, None


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
    score, score_key = extract_score(val_payload)
    score_str = "NA" if score is None else f"{score:.6f}"
    key_str = "NA" if score_key is None else score_key
    print(f"[{idx}/{total}] done group={group} score={score_str} key={key_str}", flush=True)
    all_rows[group].append((score, f, score_key))

    if score is None:
        continue
    if group not in best or score > best[group][0]:
        best[group] = (score, f, score_key)

print("BEST PER CATEGORY")
for group in sorted(best):
    score, f, score_key = best[group]
    print(f"{group}\t{score:.6f}\t{score_key}\t{f}")

print("\nTOP RUNS PER CATEGORY")
for group in sorted(k for k in all_rows if k != "FAILED"):
    print(f"\n[{group}]")
    rows = sorted(all_rows[group], key=lambda x: (-1 if x[0] is None else -x[0], x[1]))
    for score, f, score_key in rows[:10]:
        score_str = "NA" if score is None else f"{score:.6f}"
        key_str = "NA" if score_key is None else score_key
        print(f"{score_str}\t{key_str}\t{f}")

if "FAILED" in all_rows:
    print("\n[FAILED]")
    for _, f, err in all_rows["FAILED"]:
        print(f"{f}\t{err}")
PY
