# CREMA-D

Use this folder to train/evaluate CREMA-D experiments.

- Train: `./run/crema_d/train.sh [scenario|config_path] [extra args]`
- Show/evaluate: `./run/crema_d/show.sh [scenario|config_path] [extra args]`
- Config root: `run/configs/CREMA_D/`

## Scenarios

`train.sh` and `show.sh` support named scenarios (for readable, reproducible settings).

List them with:

```bash
./run/crema_d/train.sh --scenarios
./run/crema_d/show.sh --scenarios
```

The same scenario name can be used for both training and evaluation so you do not have to manually retype `--perturb`, `--l`, `--perturb_pmin`, or `--perturb_lsparse`.

## Typical workflow

```bash
# 1) train a scenario
./run/crema_d/train.sh rmask-random-l1.0-pmin0.20 --fold 0

# 2) evaluate exactly the same scenario
./run/crema_d/show.sh rmask-random-l1.0-pmin0.20 --fold 0
```

## More examples

```bash
./run/crema_d/train.sh
./run/crema_d/train.sh rmask-random-l1.0-pmin0.20 --fold 0
./run/crema_d/train.sh rmask-learned-l1.0-lsparse0.010 --fold 0
./run/crema_d/show.sh rmask-random-l1.0-pmin0.20 --fold 0
./run/crema_d/show.sh run/configs/CREMA_D/release/res/MCR.json --fold 1
```
