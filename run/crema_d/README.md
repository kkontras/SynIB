# CREMA-D

- Configs (single location): `run/configs/`
- Train: `./train.sh [scenario|config_path] [extra train args]`
- Evaluate/Show: `./show.sh [scenario|config_path] [extra show args]`
- List explicit scenarios: `./train.sh --scenarios` or `./show.sh --scenarios`

Examples:

```bash
./run/crema_d/train.sh
./run/crema_d/train.sh rmask-random-l1.0-pmin0.20 --fold 0
./run/crema_d/train.sh rmask-learned-l1.0-lsparse0.010 --fold 0
./run/crema_d/show.sh rmask-random-l1.0-pmin0.20 --fold 0
./run/crema_d/show.sh run/configs/CREMA_D/release/res/MCR.json --fold 1
```
