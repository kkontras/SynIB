# ESNLI

Use these wrappers for e-SNLI training and evaluation.

- Train: `./run/esnli/train.sh [config_path] [extra args]`
- Show/evaluate: `./run/esnli/show.sh [config_path] [extra args]`
- Config root: `run/configs/ESNLI/`

## Typical workflow

```bash
# 1) train
./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --lr 1e-4

# 2) evaluate the same config
./run/esnli/show.sh run/configs/ESNLI/cache_synib_lora.json
```

## More examples

```bash
./run/esnli/train.sh
./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --lr 1e-4
./run/esnli/show.sh run/configs/ESNLI/cache_ens.json
```

## Tip

To view all available e-SNLI configs:

```bash
./run/list_configs.sh esnli
```
