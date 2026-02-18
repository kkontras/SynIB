# ESNLI

- Configs (single location): `run/configs/`
- Train: `./train.sh [config_path] [extra train args]`
- Evaluate/Show: `./show.sh [config_path] [extra show args]`

Examples:

```bash
./run/esnli/train.sh
./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --lr 1e-4
./run/esnli/show.sh run/configs/ESNLI/cache_ens.json
```
