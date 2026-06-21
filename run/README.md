# Run scripts

Thin wrappers around the Python entrypoints (`synib.entrypoints.train` /
`synib.entrypoints.show`). Each wrapper sets `PYTHONPATH=src` and `cd`s to the repo
root, then forwards any extra CLI flags to the entrypoint.

## Per-dataset launchers

```bash
# MOSI / UR-Funny / MUStARD  (frozen MultiBench V+T features)
./run/multibench/train.sh <target> <config.json> [--fold N] [--rmask random|learned] [--l λ] ...
./run/multibench/show.sh  <target> <config.json> [--fold N]
#   targets: mosi-vt | urfunny-vt | mustard-vt

# Hateful Memes  (CLIP-ViT + DeBERTa frozen features)
./run/hateful_memes/train.sh <tier.json> [method.json] [--fold N] ...

# CREMA-D / CREMA-D-Irony  (audio + visual)
./run/cremad/train.sh [--rmask random|learned] [--l λ] [--pmin p] [--fold N] ...
./run/cremad/show.sh  ... (same flags)
```

See the dataset-specific READMEs for targets, scenarios and full examples:
`run/multibench/README.md`, `run/cremad/README.md`.

## Configs

All experiment configs live under `run/configs/`:

```text
run/configs/
  multibench/{mosi,urfunny,mustard}/   MOSI, UR-Funny, MUStARD
  hateful_memes/{tiers,methods}/       HM backbone tiers x method overlays
  cremad/                              CREMA-D / irony
```

## Overrides

Extra flags pass straight through to the entrypoint:

```bash
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json \
                  run/configs/hateful_memes/methods/synib.json --fold 0 --lr 1e-4 --wd 1e-5
```

> Configs use `./data/...` path placeholders — edit the `default_config_*.json` for each
> dataset to point at your local data and checkpoint directories.
