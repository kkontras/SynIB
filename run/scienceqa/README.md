# ScienceQA

- Configs (single location): `run/configs/`
- Train: `./train.sh [config_path] [extra train args]`
- Evaluate/Show: `./show.sh [config_path] [extra show args]`

Examples:

```bash
./run/scienceqa/train.sh
./run/scienceqa/train.sh run/configs/ScienceQA/cache_synib_lora.json --lr 1e-4
./run/scienceqa/show.sh run/configs/ScienceQA/cache_lora_MCR.json
```
