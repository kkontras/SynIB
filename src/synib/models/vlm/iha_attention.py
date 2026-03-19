"""Interleaved Head Attention (IHA-Lite) — pure PyTorch, no SynIB deps.

Adds three tiny mixing matrices (M_Q, M_K, M_V) per attention layer that blend
Q/K/V across heads before attention.  Identity init → exact MHA at start.

Reference: arxiv 2602.21371
"""
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class IHAMixingLayer(nn.Module):
    """Learnable head-mixing for Q, K, V.

    M_Q  : [H_q,  P_q]   — blends query heads
    M_K  : [H_kv, P_kv]  — blends key heads
    M_V  : [H_kv, P_kv]  — blends value heads

    Identity init (H == P) recovers exact MHA.
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        num_pseudo_q: int = None,
        num_pseudo_kv: int = None,
        init: str = "identity",
        noise_std: float = 0.01,
    ):
        super().__init__()
        P_q = num_pseudo_q if num_pseudo_q is not None else num_q_heads
        P_kv = num_pseudo_kv if num_pseudo_kv is not None else num_kv_heads

        self.M_Q = nn.Parameter(self._init_matrix(num_q_heads, P_q, init, noise_std))
        self.M_K = nn.Parameter(self._init_matrix(num_kv_heads, P_kv, init, noise_std))
        self.M_V = nn.Parameter(self._init_matrix(num_kv_heads, P_kv, init, noise_std))

    @staticmethod
    def _init_matrix(H: int, P: int, init: str, noise_std: float) -> torch.Tensor:
        if init == "identity":
            if H == P:
                m = torch.eye(H, P)
            else:
                m = nn.init.orthogonal_(torch.empty(H, P))
        elif init == "identity_noise":
            if H == P:
                m = torch.eye(H, P)
            else:
                m = nn.init.orthogonal_(torch.empty(H, P))
            m = m + noise_std * torch.randn_like(m)
        elif init == "orthogonal":
            m = nn.init.orthogonal_(torch.empty(H, P))
        else:
            raise ValueError(f"Unknown IHA init: {init!r}")
        return m

    def forward(
        self,
        query_states: torch.Tensor,   # [B, H_q,  N, d_h]
        key_states: torch.Tensor,     # [B, H_kv, N, d_h]
        value_states: torch.Tensor,   # [B, H_kv, N, d_h]
    ):
        dtype = query_states.dtype
        M_Q  = self.M_Q.to(dtype)
        M_K  = self.M_K.to(dtype)
        M_V  = self.M_V.to(dtype)

        # bhnd,hp->bpnd
        Q = torch.einsum("bhnd,hp->bpnd", query_states, M_Q)
        K = torch.einsum("bhnd,hp->bpnd", key_states,   M_K)
        V = torch.einsum("bhnd,hp->bpnd", value_states, M_V)
        return Q, K, V


def patch_attention_with_iha(
    attn_module,
    num_q_heads: int,
    num_kv_heads: int,
    num_pseudo_q: int = None,
    num_pseudo_kv: int = None,
    init: str = "identity",
    noise_std: float = 0.01,
):
    """Monkey-patch a single Qwen3VLTextAttention with IHA mixing."""
    mixing = IHAMixingLayer(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_pseudo_q=num_pseudo_q,
        num_pseudo_kv=num_pseudo_kv,
        init=init,
        noise_std=noise_std,
    )
    attn_module.iha_mixing = mixing

    def iha_forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)

        query_states = attn_module.q_norm(attn_module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states   = attn_module.k_norm(attn_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ── IHA MIXING (injected after proj+norm, before RoPE) ──
        query_states, key_states, value_states = attn_module.iha_mixing(
            query_states, key_states, value_states
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn_module.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if attn_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_module.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            attn_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
            scaling=attn_module.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_module.o_proj(attn_output)
        return attn_output, attn_weights

    attn_module.forward = iha_forward
    return mixing


def apply_iha_to_model(
    model,
    layers="all",
    num_pseudo_q: int = None,
    num_pseudo_kv: int = None,
    init: str = "identity",
    noise_std: float = 0.01,
):
    """Patch all target transformer layers with IHA mixing.

    Handles raw HF backbone and PEFT-wrapped variants.
    Returns a list of IHAMixingLayer instances (one per patched layer).
    """
    # Navigate to language_model.layers — try multiple PEFT wrapping depths
    transformer_layers = None
    for attr_path in [
        "base_model.model.model.language_model.layers",   # PEFT double-wrap
        "base_model.model.language_model.layers",          # PEFT single-wrap
        "model.language_model.layers",                     # raw HF
    ]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            transformer_layers = obj
            break
        except AttributeError:
            continue

    if transformer_layers is None:
        raise RuntimeError("Could not locate language_model.layers in model")

    num_layers = len(transformer_layers)
    if layers == "all":
        target_indices = list(range(num_layers))
    else:
        target_indices = [int(i) for i in layers]

    # Read head counts from config of first target layer's self_attn
    first_attn = transformer_layers[target_indices[0]].self_attn
    num_q_heads  = int(first_attn.config.num_attention_heads)
    num_kv_heads = int(first_attn.config.num_key_value_heads)

    mixing_layers = []
    for idx in target_indices:
        attn = transformer_layers[idx].self_attn
        mixing = patch_attention_with_iha(
            attn,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            num_pseudo_q=num_pseudo_q,
            num_pseudo_kv=num_pseudo_kv,
            init=init,
            noise_std=noise_std,
        )
        mixing_layers.append(mixing)

    return mixing_layers
