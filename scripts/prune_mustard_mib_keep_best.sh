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


def extract_combined_acc(payload):
    acc = payload.get("acc")
    if isinstance(acc, dict):
        for key in ("combined", "total"):
            v = acc.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        nums = [v for v in acc.values() if isinstance(v, (int, float))]
        if nums:
            return max(nums)
    for key in ("accuracy", "acc", "best_accuracy", "best_acc"):
        v = payload.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def human(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024


files = sorted(f for f in os.listdir(".") if f.endswith(".pth.tar"))
rows = defaultdict(list)

print(f"Scanning {len(files)} checkpoint files in {os.getcwd()}\n", flush=True)

for i, f in enumerate(files, start=1):
    print(f"[{i}/{len(files)}] opening {f}", flush=True)
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[{i}/{len(files)}] failed: {e}", flush=True)
        continue

    group = group_name(f)
    val_payload = extract_val_payload(ckpt)
    test_payload = extract_test_payload(ckpt, val_payload)
    val = extract_combined_acc(val_payload)
    test = extract_combined_acc(test_payload)
    size = os.path.getsize(f)
    rows[group].append((f, val, test, size))
    print(f"[{i}/{len(files)}] done group={group} val_acc_combined={val} test_acc_combined={test}", flush=True)


keep = set()

print("\nWinners per category\n", flush=True)
for group in sorted(rows):
    items = rows[group]
    vals = [x[1] for x in items if x[1] is not None]
    tests = [x[2] for x in items if x[2] is not None]

    best_val = max(vals) if vals else None
    best_test = max(tests) if tests else None

    print(f"[{group}] best_val={best_val} best_test={best_test}", flush=True)

    for f, val, test, _ in items:
        if best_val is not None and val == best_val:
            keep.add(f)
        if best_test is not None and test == best_test:
            keep.add(f)


delete = []
delete_bytes = 0
keep_bytes = 0

for group in rows:
    for f, val, test, size in rows[group]:
        if f in keep:
            keep_bytes += size
        else:
            delete.append(f)
            delete_bytes += size


print(f"\nKeeping {len(keep)} files, total {human(keep_bytes)}", flush=True)
print(f"Deleting {len(delete)} files, freeing about {human(delete_bytes)}\n", flush=True)

for f in sorted(keep):
    print(f"KEEP   {f}", flush=True)

print("", flush=True)

for f in sorted(delete):
    print(f"DELETE {f}", flush=True)

print("\nDeleting now...\n", flush=True)

for f in delete:
    os.remove(f)

print(f"Done. Deleted {len(delete)} files and freed about {human(delete_bytes)}.", flush=True)
PY
