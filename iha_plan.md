# IHA Implementation Plan for Qwen3-VL-2B — SynIB Project

> **Scope**: Model code + configs + train.py CLI args. No training loop changes beyond argparse.
> Target: MUStARD Cache pipeline first, then multi-bench.

---

## 1. Paper Recap (Interleaved Head Attention, arxiv 2602.21371)

Standard MHA: H heads compute H independent attention matrices with zero cross-head communication. IHA adds **three tiny mixing matrices per layer** (M_Q, M_K, M_V) that blend Q/K/V across heads *before* attention, creating "pseudo-heads" — each a learned linear combination of all original heads. This yields up to P² attention patterns per head (vs 1 in MHA).

**Posthoc property**: Initialize M_Q = M_K = M_V = Identity → model starts as exact vanilla MHA → fine-tune only the mixing params.

---

## 2. Mathematical Formulation

### Qwen3-VL-2B Constants
```
H_q  = 16          # query heads
H_kv = 8           # key/value heads (GQA)
d_h  = 128         # head dimension
d    = 2048         # hidden size
L    = 28           # transformer layers
```

### Standard Qwen3-VL Attention Forward (`Qwen3VLTextAttention.forward`, line 460 of modeling_qwen3_vl.py)
```python
query_states = q_norm(q_proj(hidden_states)).view(B, N, H_q, d_h).transpose(1,2)   # [B, H_q, N, d_h]
key_states   = k_norm(k_proj(hidden_states)).view(B, N, H_kv, d_h).transpose(1,2)  # [B, H_kv, N, d_h]
value_states = v_proj(hidden_states).view(B, N, H_kv, d_h).transpose(1,2)          # [B, H_kv, N, d_h]

cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
# GQA expand + attention + o_proj
```

### IHA Mixing (injected AFTER projection+norm, BEFORE RoPE)

**New learnable parameters per layer:**
```
M_Q  ∈ ℝ^{H_q  × P_q}     # [16 × P_q]    query mixing
M_K  ∈ ℝ^{H_kv × P_kv}    # [8  × P_kv]   key mixing
M_V  ∈ ℝ^{H_kv × P_kv}    # [8  × P_kv]   value mixing
```

**Default**: P_q = H_q = 16, P_kv = H_kv = 8 (preserves GQA structure).
**Configurable**: `--pseudo_heads_q` and `--pseudo_heads_kv` CLI args.

**Init**: Identity (recovers exact MHA). Can also use `identity_noise` for slight perturbation.

**Mixing operation:**
```python
Q_mixed = einsum('bhnd,hp->bpnd', Q, M_Q)   # [B, P_q, N, d_h]
K_mixed = einsum('bhnd,hp->bpnd', K, M_K)   # [B, P_kv, N, d_h]
V_mixed = einsum('bhnd,hp->bpnd', V, M_V)   # [B, P_kv, N, d_h]
```

### Variant A — IHA-Lite (recommended first)
- Mixing only, no sequence interleaving
- Same sequence length N, same attention masks, same KV cache
- Preserves GQA shape when P_q = H_q, P_kv = H_kv
- Only overhead: three small einsum ops per layer

### Variant B — Full IHA (optional follow-up)
- Mixing + interleave pseudo-heads into sequence dim: N → N·P
- Virtual RoPE positions: pos(n,p) = n·P + p
- P× attention cost and KV cache
- More expressive but much heavier

### Parameter Count
| Method | Params/layer | Total (28 layers) |
|--------|-------------|-------------------|
| IHA-Lite (default P) | 16² + 8² + 8² = 384 | **10,752** |
| LoRA (r=8, q+v) | 2 × 2 × 2048 × 8 = 65,536 | 1,835,008 |
| IHA + LoRA | 65,920 | **1,845,760** |

---

## 3. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/synib/models/vlm/iha_attention.py` | **CREATE** | Standalone IHAMixingLayer + patching functions |
| `src/synib/models/vlm/qwen_iha.py` | **CREATE** | IHA model classes extending `_QwenVL_CachedCombinedImpl` |
| `src/synib/models/vlm/__init__.py` | **MODIFY** | Add `from .qwen_iha import *` |
| `src/synib/entrypoints/train.py` | **MODIFY** | Add `--pseudo_heads_q`, `--pseudo_heads_kv`, `--iha_init`, `--iha_lr`, `--iha_layers` args |
| `run/configs/MUStARD/cache_iha.json` | **CREATE** | IHA-only config (no LoRA) |
| `run/configs/MUStARD/cache_iha_lora.json` | **CREATE** | IHA + LoRA combined config |

---

## 4. File Specifications

### 4.1 `src/synib/models/vlm/iha_attention.py`

Core IHA module. No SynIB dependencies — pure PyTorch.

**Classes/functions to implement:**

1. **`IHAMixingLayer(nn.Module)`** — The core mixing module
   - `__init__(num_q_heads, num_kv_heads, num_pseudo_q=None, num_pseudo_kv=None, init="identity", noise_std=0.01)`
   - P_q defaults to H_q, P_kv defaults to H_kv
   - Creates M_Q [H_q, P_q], M_K [H_kv, P_kv], M_V [H_kv, P_kv] as nn.Parameter
   - Init: eye for identity (when H==P), orthogonal otherwise, optional noise
   - `forward(query_states, key_states, value_states)` → einsum mixing, cast M to input dtype, return mixed Q/K/V

2. **`patch_attention_with_iha(attn_module, ...)`** — Monkey-patches a single `Qwen3VLTextAttention`
   - Creates IHAMixingLayer, attaches as `attn_module.iha_mixing`
   - Replaces `attn_module.forward` with `iha_forward`

3. **`apply_iha_to_model(model, layers="all", ...)`** — Patches all target layers
   - Navigates to `language_model.layers` (handles PEFT wrapping)
   - Returns list of IHAMixingLayer modules

**Critical: the patched `iha_forward` must replicate `Qwen3VLTextAttention.forward` exactly:**

```python
def iha_forward(hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_module.head_dim)

    query_states = attn_module.q_norm(attn_module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states   = attn_module.k_norm(attn_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # ──── IHA MIXING (the ONLY new thing) ────
    query_states, key_states, value_states = attn_module.iha_mixing(
        query_states, key_states, value_states
    )
    # ─────────────────────────────────────────

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, attn_module.layer_idx, cache_kwargs)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn_module.config._attn_implementation, eager_attention_forward)

    attn_output, attn_weights = attention_interface(
        attn_module, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
        scaling=attn_module.scaling, **kwargs)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_module.o_proj(attn_output)
    return attn_output, attn_weights
```

**PEFT model navigation** — `apply_iha_to_model` must try multiple paths since LoRA wraps the model:
```python
# PEFT double wrap: model.base_model.model.model.language_model.layers
# Raw HF:           model.model.language_model.layers
# PEFT single wrap: model.base_model.model.language_model.layers
```

### 4.2 `src/synib/models/vlm/qwen_iha.py`

Model classes following the existing SynIB pattern. Extends `_QwenVL_CachedCombinedImpl` from `qwen_cached.py`.

**`_QwenVL_CachedIHAImpl`** — Main impl class:
- `__init__` calls `super().__init__()` (loads backbone, applies LoRA, CLS, trainables), then `_apply_iha()`, then `_setup_iha_trainables()`
- `_apply_iha()` reads `args.iha_config` dict, calls `apply_iha_to_model()`
- `_setup_iha_trainables()` marks M_Q/M_K/M_V as `requires_grad=True`
- `get_iha_parameters()` returns list of IHA params (for separate optimizer group)
- `get_iha_state_dict()` / `load_iha_state_dict()` for checkpoint save/load

**Config keys** in `args.iha_config` (dict):
```
use_iha:       bool       (default True)
layers:        "all"|list  (which layers to patch)
num_pseudo_q:  int|null    (default H_q=16)
num_pseudo_kv: int|null    (default H_kv=8)
init:          str         ("identity"|"identity_noise"|"orthogonal")
noise_std:     float       (0.01)
iha_lr:        float       (separate LR for IHA params)
```

**Public aliases** (following `qwen_cached.py` convention):
```python
class QwenVL_Cached_IHA(_QwenVL_CachedIHAImpl): pass       # IHA only
class QwenVL_Cached_IHA_LoRA(_QwenVL_CachedIHAImpl): pass   # IHA + LoRA
```

**Order of operations in `__init__`:**
```
super().__init__()
  → loads backbone (Qwen3VLForConditionalGeneration)
  → resizes embeddings (CLS token)
  → _apply_lora()      ← PEFT wraps q_proj/v_proj
  → _load_cls_embedding()
  → _setup_trainables() ← freezes backbone, unfreezes LoRA + enc_0
_apply_iha()            ← monkey-patches attention with mixing (NEW)
_setup_iha_trainables() ← marks M_Q/M_K/M_V as trainable (NEW)
```

### 4.3 `src/synib/models/vlm/__init__.py` — Add one line

```python
from .qwen_iha import *
```

### 4.4 `src/synib/entrypoints/train.py` — Modifications

**Add argparse entries** (after line ~243, before `parser.set_defaults`):

```python
# IHA arguments
parser.add_argument('--pseudo_heads_q', required=False, type=int, default=None,
                    help="IHA pseudo-heads for Q (default: num_q_heads=16)")
parser.add_argument('--pseudo_heads_kv', required=False, type=int, default=None,
                    help="IHA pseudo-heads for KV (default: num_kv_heads=8)")
parser.add_argument('--iha_init', required=False, default=None,
                    help="IHA init: identity, identity_noise, orthogonal")
parser.add_argument('--iha_lr', required=False, type=float, default=None,
                    help="Separate learning rate for IHA mixing params")
parser.add_argument('--iha_layers', required=False, default=None,
                    help="IHA layers: 'all' or comma-sep e.g. '20,21,22,23,24,25,26,27'")
```

**Add config injection** in `main()` (after the existing arg blocks, ~line 185):

```python
    # ── IHA config injection ──
    if getattr(args, "pseudo_heads_q", None) is not None:
        if not hasattr(config.model.args, "iha_config"):
            config.model.args.iha_config = {}
        config.model.args.iha_config["num_pseudo_q"] = int(args.pseudo_heads_q)
        m += "_phq{}".format(args.pseudo_heads_q)
    if getattr(args, "pseudo_heads_kv", None) is not None:
        if not hasattr(config.model.args, "iha_config"):
            config.model.args.iha_config = {}
        config.model.args.iha_config["num_pseudo_kv"] = int(args.pseudo_heads_kv)
        m += "_phkv{}".format(args.pseudo_heads_kv)
    if getattr(args, "iha_init", None) is not None:
        if not hasattr(config.model.args, "iha_config"):
            config.model.args.iha_config = {}
        config.model.args.iha_config["init"] = args.iha_init
        m += "_ihainit{}".format(args.iha_init)
    if getattr(args, "iha_lr", None) is not None:
        if not hasattr(config.model.args, "iha_config"):
            config.model.args.iha_config = {}
        config.model.args.iha_config["iha_lr"] = float(args.iha_lr)
        m += "_ihalr{}".format(args.iha_lr)
    if getattr(args, "iha_layers", None) is not None:
        if not hasattr(config.model.args, "iha_config"):
            config.model.args.iha_config = {}
        if args.iha_layers == "all":
            config.model.args.iha_config["layers"] = "all"
        else:
            config.model.args.iha_config["layers"] = [int(x) for x in args.iha_layers.split(",")]
        m += "_ihaL{}".format(args.iha_layers.replace(",", "-"))
```

---

## 5. Config Files

### 5.1 `run/configs/MUStARD/cache_iha.json` — IHA Only

```json
{
  "dataset": {
    "data_split": { "fold": 0 }
  },
  "model": {
    "model_class": "QwenVL_Cached_IHA",
    "load_ongoing": false,
    "save_dir": "MUStARD_Cache_IHA_{}.pth.tar",
    "args": {
      "d_model": 512,
      "num_classes": 2,
      "fc_inner": 64,
      "dropout": 0.1,
      "cls_finetune": false,
      "cls_type": "linear",
      "clip_grad": true,
      "clip_value": 1.0,
      "bias_infusion": { "method": "false", "use": false, "plot": false },
      "lora_config": { "use_lora": false },
      "iha_config": {
        "use_iha": true,
        "layers": "all",
        "init": "identity",
        "noise_std": 0.01,
        "iha_lr": 0.005
      },
      "multi_loss": { "multi_supervised_w": { "combined": 0, "c": 0, "g": 0 } }
    },
    "encoders": [
      {
        "model": "LinearHead_Qwen",
        "args": { "d_model": 2048, "num_classes": 2 },
        "pretrainedEncoder": { "use": false, "dir": "" }
      }
    ]
  }
}
```

### 5.2 `run/configs/MUStARD/cache_iha_lora.json` — IHA + LoRA

```json
{
  "dataset": {
    "data_split": { "fold": 0 }
  },
  "model": {
    "model_class": "QwenVL_Cached_IHA_LoRA",
    "load_ongoing": false,
    "save_dir": "MUStARD_Cache_IHA_LoRA_{}.pth.tar",
    "args": {
      "d_model": 512,
      "num_classes": 2,
      "fc_inner": 64,
      "dropout": 0.1,
      "cls_finetune": false,
      "cls_type": "linear",
      "clip_grad": true,
      "clip_value": 1.0,
      "bias_infusion": { "method": "false", "use": false, "plot": false },
      "lora_config": {
        "use_lora": true,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_bias": "none",
        "lora_lr": 0.0002
      },
      "iha_config": {
        "use_iha": true,
        "layers": "all",
        "init": "identity",
        "noise_std": 0.01,
        "iha_lr": 0.005
      },
      "multi_loss": { "multi_supervised_w": { "combined": 0, "c": 0, "g": 0 } }
    },
    "encoders": [
      {
        "model": "LinearHead_Qwen",
        "args": { "d_model": 2048, "num_classes": 2 },
        "pretrainedEncoder": { "use": false, "dir": "" }
      }
    ]
  }
}
```

Both use `--default_config run/configs/MUStARD/default_config_mustard_cache.json`.

---

## 6. Launch Examples

### IHA only — Mustard Cache
```bash
# Basic
./run/mustard/train.sh run/configs/MUStARD/cache_iha.json \
  --default_config run/configs/MUStARD/default_config_mustard_cache.json \
  --fold 0 --lr 0.001 --wd 0.01 --validate_with accuracy

# Custom pseudo-heads (P_q=8 instead of 16)
./run/mustard/train.sh run/configs/MUStARD/cache_iha.json \
  --default_config run/configs/MUStARD/default_config_mustard_cache.json \
  --fold 0 --lr 0.001 --wd 0.01 --pseudo_heads_q 8 --pseudo_heads_kv 4

# Custom IHA learning rate
./run/mustard/train.sh run/configs/MUStARD/cache_iha.json \
  --default_config run/configs/MUStARD/default_config_mustard_cache.json \
  --fold 0 --lr 0.001 --iha_lr 0.01

# IHA on last 8 layers only
./run/mustard/train.sh run/configs/MUStARD/cache_iha.json \
  --default_config run/configs/MUStARD/default_config_mustard_cache.json \
  --fold 0 --lr 0.001 --iha_layers "20,21,22,23,24,25,26,27"
```

### IHA + LoRA — Mustard Cache
```bash
./run/mustard/train.sh run/configs/MUStARD/cache_iha_lora.json \
  --default_config run/configs/MUStARD/default_config_mustard_cache.json \
  --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy
```

---

## 7. Sweep Phase for `tier1_mustard_cache.sh`

Add after existing methods phase:

```bash
IHA_CFG="run/configs/MUStARD/cache_iha.json"
IHA_LORA_CFG="run/configs/MUStARD/cache_iha_lora.json"
IHA_LRS=(0.001 0.005 0.01)
IHA_PSEUDO_QS=(8 16)

phase_iha() {
  echo "=== Phase: IHA experiments ==="
  for fold in "${FOLDS[@]}"; do
    for iha_lr in "${IHA_LRS[@]}"; do
      for phq in "${IHA_PSEUDO_QS[@]}"; do
        run_train_safe "${IHA_CFG}" \
          --fold "${fold}" --lr 0.0005 --wd 0.01 \
          --iha_lr "${iha_lr}" --pseudo_heads_q "${phq}" \
          --default_config "${DEFAULT_CONFIG}" --validate_with accuracy
      done
    done
    # IHA + LoRA
    for iha_lr in "${IHA_LRS[@]}"; do
      run_train_safe "${IHA_LORA_CFG}" \
        --fold "${fold}" --lr 0.0005 --wd 0.01 \
        --iha_lr "${iha_lr}" \
        --default_config "${DEFAULT_CONFIG}" --validate_with accuracy
    done
  done
}
```

Add `iha` to the MODE case statement.

---

## 8. Optimizer Integration

The existing `Loader.py` uses `self.agent.model.parameters()` which picks up all `requires_grad=True` params. IHA params will get the global LR by default — this works for initial testing.

For **separate LR** (follow-up), the model exposes `get_iha_parameters()`. Add ~10 lines to `Loader.load_models_n_optimizer()`:

```python
if hasattr(self.agent.model, 'get_iha_parameters'):
    iha_params = list(self.agent.model.get_iha_parameters())
    iha_lr = getattr(self.agent.config.model.args, "iha_config", {}).get("iha_lr", None)
    if iha_params and iha_lr:
        iha_ids = {id(p) for p in iha_params}
        other = [p for p in self.agent.model.parameters() if p.requires_grad and id(p) not in iha_ids]
        param_groups = [{'params': other}, {'params': iha_params, 'lr': iha_lr}]
        # use param_groups instead of model.parameters() in optimizer init
```

---

## 9. Multi-Bench Extension

Once Mustard works, extending is config-only (copy + edit num_classes + save_dir):

| Benchmark | num_classes | default_config |
|-----------|------------|----------------|
| MUStARD | 2 | `default_config_mustard_cache.json` |
| ScienceQA | 5 | `default_config_scienceqa_cache.json` |
| ESNLI | 3 | `default_config_esnli_cache.json` |

---

## 10. Checklist for Coding Agent

- [ ] **Create** `src/synib/models/vlm/iha_attention.py` — `IHAMixingLayer`, `patch_attention_with_iha`, `apply_iha_to_model`
- [ ] **Create** `src/synib/models/vlm/qwen_iha.py` — `_QwenVL_CachedIHAImpl`, `QwenVL_Cached_IHA`, `QwenVL_Cached_IHA_LoRA`
- [ ] **Edit** `src/synib/models/vlm/__init__.py` — add `from .qwen_iha import *`
- [ ] **Edit** `src/synib/entrypoints/train.py` — 5 argparse entries + config injection block in `main()`
- [ ] **Create** `run/configs/MUStARD/cache_iha.json`
- [ ] **Create** `run/configs/MUStARD/cache_iha_lora.json`
- [ ] **Test**: identity-init IHA output == baseline output (numerical match)
- [ ] **Test**: `--pseudo_heads_q 8` creates M_Q of shape [16, 8]
- [ ] **Test**: IHA params show `requires_grad=True`, backbone params show `False`
- [ ] **Test**: full forward+backward on Mustard cache data

### Key constraints
- IHA mixing: AFTER q/k/v projection+norm, BEFORE RoPE
- Identity init = exact MHA recovery (zero degradation at start)
- Patched forward replicates `Qwen3VLTextAttention.forward` exactly except for mixing insert
- Must work with both PEFT-wrapped and raw backbone
- IHA params in fp32, cast to input dtype for einsum
- P_q = H_q = 16, P_kv = H_kv = 8 by default (overridable via CLI)