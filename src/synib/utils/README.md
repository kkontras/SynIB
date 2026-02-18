# utils package

This folder contains shared utilities used across training, evaluation, and analysis.

## Organized layout
- `configuration/`
  - `config.py`: config loading/merging and logger setup.
  - `dirs.py`: directory creation helpers.
- `data/`
  - `to_device.py`: recursive tensor/device casting helpers.
  - `flattendict.py`: flatten nested loss dicts for logging.
- `system/`
  - `deterministic.py`: reproducibility helper (`deterministic`).
  - `misc.py`: timing decorator and CUDA stats printer.
- `metrics/`
  - `corr_metrics.py`: feature-space comparison and t-SNE plotting helpers.
- `optimization/`
  - `normalized_adamw.py`: custom optimizer (`NormalizedAdamW`).
  - `min_norm_solver.py`: min-norm multi-objective solver.
  - `gs_plugin.py`: gradient surgery helper.
- `schedulers/`
  - `no_scheduler.py`, `warmup_scheduler.py`

## Compatibility
Legacy module paths remain available:
- `utils/config.py`, `utils/to_device.py`, `utils/MinNormSolver.py`, etc.

These are thin wrappers that re-export from the new organized modules, so existing imports continue to work.
