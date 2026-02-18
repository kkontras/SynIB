# Run Scripts

This folder contains small wrappers around the Python entrypoints:

- `synib.entrypoints.train`
- `synib.entrypoints.show`

Each dataset folder (`crema_d`, `scienceqa`, `esnli`) has its own `train.sh` and `show.sh`.

## Quick start

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

## Browse available configs

```bash
./run/list_configs.sh all
./run/list_configs.sh crema_d
./run/list_configs.sh scienceqa
./run/list_configs.sh esnli
```

## Passing overrides

All scripts forward extra CLI flags directly to the Python entrypoint, so overrides are easy:

```bash
./run/scienceqa/train.sh run/configs/ScienceQA/cache_synib_lora.json --lr 1e-4 --wd 1e-5
./run/esnli/show.sh run/configs/ESNLI/cache_ens.json --fold 0
```

## Notes

- Use dataset-specific READMEs for scenario names and examples:
  - `run/crema_d/README.md`
  - `run/scienceqa/README.md`
  - `run/esnli/README.md`
