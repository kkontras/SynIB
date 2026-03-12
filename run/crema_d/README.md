# CREMA-D

Use this folder to train/evaluate CREMA-D experiments.

- Train: `./run/crema_d/train.sh [scenario|config_path] [extra args]`
- Show/evaluate: `./run/crema_d/show.sh [scenario|config_path] [extra args]`
- Config root: `run/configs/CREMA_D/`

## Scenarios

`train.sh` and `show.sh` support named scenarios, but the preferred usage is to pass explicit flags so the command itself shows the actual setup.

List them with:

```bash
./run/crema_d/train.sh --scenarios
./run/crema_d/show.sh --scenarios
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
./run/crema_d/train.sh --rmask random --l 1.0 --pmin 0.20 --fold 0

# 2) evaluate exactly the same setup
./run/crema_d/show.sh --rmask random --l 1.0 --pmin 0.20 --fold 0
```

## More examples

```bash
./run/crema_d/train.sh
./run/crema_d/train.sh --rmask random --l 1.0 --pmin 0.20 --fold 0
./run/crema_d/train.sh --rmask learned --l 1.0 --lsparse 0.010 --fold 0
./run/crema_d/show.sh --rmask random --l 1.0 --pmin 0.20 --fold 0
./run/crema_d/show.sh run/configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 1
```
