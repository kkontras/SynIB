"""IHA model classes for Qwen3-VL — extends _QwenVL_CachedCombinedImpl.

QwenVL_Cached_IHA      — IHA mixing only (no LoRA)
QwenVL_Cached_IHA_LoRA — IHA mixing + LoRA
"""
from .qwen_cached import _QwenVL_CachedCombinedImpl
from .iha_attention import apply_iha_to_model


class _QwenVL_CachedIHAImpl(_QwenVL_CachedCombinedImpl):
    """Extends the cached Qwen backbone with Interleaved Head Attention mixing.

    Order of operations:
        super().__init__()          → loads backbone, LoRA, CLS, trainables
        _apply_iha()                → monkey-patches attention layers
        _setup_iha_trainables()     → marks M_Q/M_K/M_V as requires_grad
    """

    def __init__(self, args, encs=None, **kwargs):
        super().__init__(args, encs=encs, **kwargs)
        self._iha_mixing_layers = []
        self._apply_iha()
        self._setup_iha_trainables()

    def _apply_iha(self):
        cfg = getattr(self.args, "iha_config", None)
        if cfg is None or not cfg.get("use_iha", True):
            return

        layers     = cfg.get("layers", "all")
        pseudo_q   = cfg.get("num_pseudo_q", None)
        pseudo_kv  = cfg.get("num_pseudo_kv", None)
        init       = cfg.get("init", "identity")
        noise_std  = float(cfg.get("noise_std", 0.01))

        self._iha_mixing_layers = apply_iha_to_model(
            self.backbone,
            layers=layers,
            num_pseudo_q=pseudo_q,
            num_pseudo_kv=pseudo_kv,
            init=init,
            noise_std=noise_std,
        )

    def _setup_iha_trainables(self):
        for mixing in self._iha_mixing_layers:
            for p in mixing.parameters():
                p.requires_grad = True

    def get_iha_parameters(self):
        """Yield all IHA mixing parameters (for a separate optimizer group)."""
        for mixing in self._iha_mixing_layers:
            yield from mixing.parameters()

    def get_iha_state_dict(self):
        return {
            f"iha_layer_{i}.{k}": v
            for i, mixing in enumerate(self._iha_mixing_layers)
            for k, v in mixing.state_dict().items()
        }

    def load_iha_state_dict(self, state_dict):
        for i, mixing in enumerate(self._iha_mixing_layers):
            prefix = f"iha_layer_{i}."
            layer_sd = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            mixing.load_state_dict(layer_sd, strict=True)


class QwenVL_Cached_IHA(_QwenVL_CachedIHAImpl):
    """IHA mixing only (no LoRA)."""
    pass


class QwenVL_Cached_IHA_LoRA(_QwenVL_CachedIHAImpl):
    """IHA mixing + LoRA."""
    pass
