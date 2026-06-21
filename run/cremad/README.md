# CREMA-D

Use this folder to train/evaluate CREMA-D experiments.

- Train: `./run/cremad/train.sh [scenario|config_path] [extra args]`
- Show/evaluate: `./run/cremad/show.sh [scenario|config_path] [extra args]`
- Config root: `run/configs/cremad/`

## Scenarios

`train.sh` and `show.sh` support named scenarios, but the preferred usage is to pass explicit flags so the command itself shows the actual setup.

List them with:

```bash
./run/cremad/train.sh --scenarios
./run/cremad/show.sh --scenarios
```

Short aliases:

- `--rmask random` maps to `--perturb random --perturb_fill random`
- `--rmask learned` maps to `--perturb learned --perturb_fill learned`
- `--pmin` maps to `--perturb_pmin`
- `--pmax` maps to `--perturb_pmax`
- `--lsparse` maps to `--perturb_lsparse`

## Typical workflow

```bash
# 1) train with explicit flags
./run/cremad/train.sh --rmask random --l 1.0 --pmin 0.20 --fold 0

# 2) evaluate exactly the same setup
./run/cremad/show.sh --rmask random --l 1.0 --pmin 0.20 --fold 0
```

## More examples

```bash
./run/cremad/train.sh
./run/cremad/train.sh --rmask random --l 1.0 --pmin 0.20 --fold 0
./run/cremad/train.sh --rmask learned --l 1.0 --lsparse 0.010 --fold 0
./run/cremad/show.sh --rmask random --l 1.0 --pmin 0.20 --fold 0
./run/cremad/show.sh run/configs/cremad/default.json --fold 1
```
