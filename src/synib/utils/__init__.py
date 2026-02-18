"""Utilities public surface.

Keep explicit exports here so the package layout is predictable and easy to navigate.
"""

from .configuration.config import process_config, process_config_default, setup_logger
from .configuration.dirs import create_dirs
from .metrics.corr_metrics import compare_feature_spaces, print_feature_comparisons, tsne_plot
from .system.deterministic import deterministic
from .data.flattendict import flatten_loss_dict
from .system.misc import print_cuda_statistics, timeit
from .data.to_device import to_device, to_float
from .optimization.gs_plugin import GSPlugin
from .optimization.min_norm_solver import MinNormSolver
try:
    from .optimization.normalized_adamw import NormalizedAdamW
except Exception:
    NormalizedAdamW = None

__all__ = [
    "GSPlugin",
    "MinNormSolver",
    "NormalizedAdamW",
    "compare_feature_spaces",
    "create_dirs",
    "deterministic",
    "flatten_loss_dict",
    "print_cuda_statistics",
    "print_feature_comparisons",
    "process_config",
    "process_config_default",
    "setup_logger",
    "timeit",
    "to_device",
    "to_float",
    "tsne_plot",
]
