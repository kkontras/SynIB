# SynIB

SynIB is the training/evaluation codebase for:

- CREMA-D
- ScienceQA
- e-SNLI

Start from the `run/` scripts. They wrap the Python entrypoints and are the easiest way to launch experiments.

## Install

```bash
pip install -r requirements.txt
```

## Quickstart (from repo root)

```bash
# CREMA-D
./run/crema_d/train.sh
./run/crema_d/show.sh

# ScienceQA
./run/scienceqa/train.sh
./run/scienceqa/show.sh

# e-SNLI
./run/esnli/train.sh
./run/esnli/show.sh
```

## Command pattern

All dataset scripts follow the same shape:

```bash
./run/<dataset>/train.sh [config_or_scenario] [extra args]
./run/<dataset>/show.sh  [config_or_scenario] [extra args]
```

Examples:

```bash
./run/crema_d/train.sh run/configs/CREMA_D/synergy/dec/synprom_RMask.json --fold 0
./run/scienceqa/train.sh run/configs/ScienceQA/cache_synib_lora.json --lr 1e-4
./run/esnli/show.sh run/configs/ESNLI/cache_ens.json --fold 0
```

## Browse configs

```bash
./run/list_configs.sh all
./run/list_configs.sh crema_d
./run/list_configs.sh scienceqa
./run/list_configs.sh esnli
```

## CREMA-D named scenarios

CREMA-D supports explicit named scenarios so train and show use identical settings.

```bash
./run/crema_d/train.sh --scenarios
./run/crema_d/show.sh --scenarios

./run/crema_d/train.sh rmask-random-l1.0-pmin0.20 --fold 0
./run/crema_d/show.sh rmask-random-l1.0-pmin0.20 --fold 0
```

## Repository map

- `run/`: user-facing shell wrappers
- `run/configs/`: experiment configs (JSON)
- `src/synib/entrypoints/`: Python CLI entrypoints (`train`, `show`)
- `src/synib/training/`: training orchestration
- `src/synib/models/`: model implementations
- `src/synib/mydatasets/`: dataset loaders/utilities

## More documentation

- `run/README.md`
- `run/crema_d/README.md`
- `run/scienceqa/README.md`
- `run/esnli/README.md`
- `src/synib/models/README.md`
