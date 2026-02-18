# SynIB

`SynIB` is organized as a proper, source-first repository for the selected workloads:
- CREMA-D
- ScienceQA
- ESNLI
- XOR

`Synergy` was not modified.

## Structure

- `src/synib/`: all source code (single source-of-truth)
  - `src/synib/training/`: training pipeline orchestration and helpers
  - `src/synib/models/`: task-organized model implementations
  - `src/synib/mydatasets/`: dataset loaders and task-specific dataset utilities
  - `src/synib/utils/`: config/system/metrics/optimization utilities
  - `src/synib/entrypoints/`: train/show entrypoint scripts
- `run/configs/`: all task configs (`CREMA_D`, `ScienceQA`, `ESNLI`, `xor`)
- `run/`: top-level execution scripts for users

## Model Navigation

See `src/synib/models/README.md`.

## Install

```bash
pip install -r requirements.txt
```

## Run (Top-level)

```bash
./run/train_crema_d.sh
./run/eval_crema_d.sh

./run/train_scienceqa.sh
./run/eval_scienceqa.sh

./run/train_esnli.sh
./run/eval_esnli.sh

./run/train_xor.sh
./run/eval_xor.sh
```

List configs:

```bash
./run/list_configs.sh all
```

Run with a specific config:

```bash
./run/train_crema_d.sh run/configs/CREMA_D/synergy/dec/synprom_RMask.json --fold 0
```
