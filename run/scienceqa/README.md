# ScienceQA

Use these wrappers for ScienceQA training and evaluation.

- Train: `./run/scienceqa/train.sh [config_path] [extra args]`
- Show/evaluate: `./run/scienceqa/show.sh [config_path] [extra args]`
- Config root: `run/configs/ScienceQA/`

## Typical workflow

```bash
# 1) train
./run/scienceqa/train.sh run/configs/ScienceQA/cache_synib_lora.json --lr 1e-4

# 2) evaluate the same config
./run/scienceqa/show.sh run/configs/ScienceQA/cache_synib_lora.json
```

## More examples

```bash
./run/scienceqa/train.sh
./run/scienceqa/train.sh run/configs/ScienceQA/cache_synib_lora.json --lr 1e-4
./run/scienceqa/show.sh run/configs/ScienceQA/cache_lora_MCR.json
```

## Tip

If you are unsure which config to start from, run:

```bash
./run/list_configs.sh scienceqa
```
