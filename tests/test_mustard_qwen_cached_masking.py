"""
Modality-masking tests for the two cached-embedding Qwen models used with MUStARD.

  - _QwenVL_CachedTextImpl  (text-only)  → must suppress vision positions
  - _QwenVL_CachedImageImpl (image-only) → must suppress text-hint positions

All tests run on CPU with no real model weights.
"""

import copy
import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D_MODEL = 8
VOCAB_SIZE = 200
IMAGE_TOKEN_ID = 42
CLS_TOKEN_ID = 99
T_SEQ = 16
N_DSV = 2        # deepstack layers
N_IMG = 4        # image token positions per sample
B = 2            # batch size


# ---------------------------------------------------------------------------
# AttrDict helper
# ---------------------------------------------------------------------------
class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


# ---------------------------------------------------------------------------
# Fake config / backbone / tokenizer / processor
# ---------------------------------------------------------------------------
class _FakeConfig:
    image_token_id = IMAGE_TOKEN_ID
    text_config = SimpleNamespace(hidden_size=D_MODEL)
    hidden_size = D_MODEL


class _FakeLM(nn.Module):
    """Small real nn.Module so embed_tokens works; also records last forward kwargs.

    When deepstack_visual_embeds is passed (CachedImage), its mean is blended into
    every token hidden state so that changes to deepstack affect the final output.
    """

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, D_MODEL)
        # Identity projection so outputs vary with inputs (for invariance / varies tests)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        nn.init.eye_(self.proj.weight)
        self.last_call_kwargs = {}

    def forward(self, **kwargs):
        self.last_call_kwargs = {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }
        ie = kwargs["inputs_embeds"]
        B_, T = ie.shape[:2]
        projected = self.proj(ie)   # (B, T, D_MODEL)

        # If deepstack is present (CachedImage path), blend its mean into every token
        dsv = kwargs.get("deepstack_visual_embeds", None)
        if dsv is not None:
            # dsv is a list of tensors (shape (B*N_IMG, D_MODEL) each)
            dsv_mean = torch.stack([d.float().mean(0) for d in dsv], dim=0).mean(0)  # (D_MODEL,)
            projected = projected + dsv_mean.unsqueeze(0).unsqueeze(0).to(projected.dtype)

        return SimpleNamespace(hidden_states=(projected,))


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self._lm = _FakeLM()
        self.model = SimpleNamespace(language_model=self._lm)
        self.generate_call_kwargs = {}

    @property
    def device(self):
        return torch.device("cpu")

    def resize_token_embeddings(self, n):
        pass

    def parameters(self):
        return iter([])

    def named_parameters(self):
        return iter([])

    def generate(self, **kwargs):
        self.generate_call_kwargs = {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }
        base = kwargs.get("inputs_embeds")
        B_ = base.shape[0]
        T_ = base.shape[1]
        return torch.zeros(B_, T_ + 3, dtype=torch.long)


class _FakeTokenizer:
    padding_side = "left"
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return VOCAB_SIZE

    def add_special_tokens(self, d):
        return 1

    def convert_tokens_to_ids(self, tok):
        return CLS_TOKEN_ID

    def batch_decode(self, ids, **kw):
        return ["ok"] * ids.shape[0]


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D_MODEL, 2)

    def forward(self, x, **kwargs):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Args factory
# ---------------------------------------------------------------------------
def _make_args():
    return AttrDict(
        num_classes=2,
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        save_base_dir=None,
        cls_emb_path=None,
        lora_config=None,
        cls_finetune=False,
    )


# ---------------------------------------------------------------------------
# Batch factory
# ---------------------------------------------------------------------------
def _make_cached_batch(batch_size=B, n_img_pos=N_IMG, n_hint_pos=2):
    T = T_SEQ
    ids = torch.full((batch_size, T), 5, dtype=torch.long)
    ids[:, 1:1 + n_img_pos] = IMAGE_TOKEN_ID
    ids[:, T - 1] = CLS_TOKEN_ID

    image_mask = torch.zeros(batch_size, T, dtype=torch.bool)
    image_mask[:, 1:1 + n_img_pos] = True

    hint_mask = torch.zeros(batch_size, T, dtype=torch.bool)
    hint_mask[:, 1 + n_img_pos:1 + n_img_pos + n_hint_pos] = True

    input_embeds = torch.rand(batch_size, T, D_MODEL)
    position_ids = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    deep_stack_viz = torch.rand(N_DSV, batch_size * n_img_pos, D_MODEL)

    return {
        "input_ids": ids,
        "input_embeds": input_embeds,
        "attention_mask": torch.ones(batch_size, T, dtype=torch.long),
        "image_mask": image_mask,
        "visual_pos_masks": image_mask,
        "hint_mask": hint_mask,
        "position_ids": position_ids,
        "deepstack_visual_embeds": deep_stack_viz,
    }


# ---------------------------------------------------------------------------
# Module loader — stubs heavy deps, patches from_pretrained
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHED_MODULE_PATH = (
    REPO_ROOT / "src" / "synib" / "models" / "vlm" / "qwen_cached.py"
)


def _build_stubs():
    """Install lightweight stubs for all heavy transitive imports."""
    import importlib.machinery

    def _stub(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = m
        return m

    # peft — stub only if not already installed as a real package
    if "peft" not in sys.modules:
        try:
            import importlib.util as _ilu
            real_spec = _ilu.find_spec("peft")
        except (ValueError, ModuleNotFoundError):
            real_spec = None
        if real_spec is None:
            peft = _stub("peft")
            peft.LoraConfig = None
            peft.get_peft_model = lambda model, cfg: model

    # wandb
    if "wandb" not in sys.modules:
        _stub("wandb")

    # einops
    if "einops" not in sys.modules:
        einops = _stub("einops")
        einops.rearrange = lambda x, *a, **kw: x

    # pytorch_metric_learning
    for pkg in ("pytorch_metric_learning", "pytorch_metric_learning.losses"):
        if pkg not in sys.modules:
            m = _stub(pkg)
            m.NTXentLoss = type("NTXentLoss", (), {"__init__": lambda self, **kw: None})

    # synib.models.model_utils + fusion_gates
    for pkg in ("synib.models.model_utils", "synib.models.model_utils.fusion_gates"):
        if pkg not in sys.modules:
            _stub(pkg)

    # synib.models (package stub so the relative imports inside qwen_cached work)
    for pkg in ("synib", "synib.models", "synib.models.vlm"):
        if pkg not in sys.modules:
            _stub(pkg)


def _load_cached_module():
    """Load qwen_cached.py with all heavy deps stubbed."""
    _build_stubs()

    # Stub the qwen_base_models imports that qwen_cached pulls in via relative import
    qbm = types.ModuleType("synib.models.vlm.qwen_base_models")
    qbm.LinearHead_Qwen = _FakeHead
    qbm.SynIB_QwenFaster = object   # not used in cached classes
    sys.modules["synib.models.vlm.qwen_base_models"] = qbm

    spec = importlib.util.spec_from_file_location(
        "synib.models.vlm.qwen_cached", CACHED_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # Set package so relative imports inside the file resolve correctly
    module.__package__ = "synib.models.vlm"
    sys.modules["synib.models.vlm.qwen_cached"] = module

    assert spec.loader is not None
    # exec_module runs `from transformers import AutoProcessor, Qwen3VLForConditionalGeneration`
    # which sets module-level names; we patch AFTER so __init__ sees our fakes.
    spec.loader.exec_module(module)

    # Patch the module-level names that __init__ will call .from_pretrained() on
    fake_backbone_cls = type(
        "Qwen3VLForConditionalGeneration",
        (),
        {"from_pretrained": staticmethod(lambda *a, **kw: _FakeBackbone())},
    )
    fake_processor_cls = type(
        "AutoProcessor",
        (),
        {"from_pretrained": staticmethod(lambda *a, **kw: _FakeProcessor())},
    )
    module.Qwen3VLForConditionalGeneration = fake_backbone_cls
    module.AutoProcessor = fake_processor_cls
    # Ensure peft symbols are identity / no-op
    module.LoraConfig = None
    module.get_peft_model = lambda model, cfg: model

    return module


# Load once at import time
_MOD = _load_cached_module()
_QwenVL_CachedTextImpl = _MOD._QwenVL_CachedTextImpl
_QwenVL_CachedImageImpl = _MOD._QwenVL_CachedImageImpl


def _make_text_model():
    head = _FakeHead()
    model = _QwenVL_CachedTextImpl(_make_args(), encs=[head])
    return model


def _make_image_model():
    head = _FakeHead()
    model = _QwenVL_CachedImageImpl(_make_args(), encs=[head])
    return model


# ---------------------------------------------------------------------------
# TestCachedTextMasking
# ---------------------------------------------------------------------------
class TestCachedTextMasking(unittest.TestCase):

    def setUp(self):
        self.model = _make_text_model()
        self.lm = self.model.backbone._lm
        self.batch = _make_cached_batch()

    def _run(self, batch=None, **kwargs):
        b = batch if batch is not None else self.batch
        return self.model.forward(b, **kwargs)

    # --- embedding masking ---

    def test_image_positions_set_to_near_zero(self):
        self._run()
        recv = self.lm.last_call_kwargs["inputs_embeds"]
        image_mask = self.batch["image_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if image_mask[b, t]:
                    self.assertTrue(
                        (recv[b, t] - 1e-5).abs().max() < 1e-7,
                        f"Position ({b},{t}) expected 1e-5, got {recv[b, t]}",
                    )

    def test_non_image_positions_unchanged(self):
        original = self.batch["input_embeds"].clone()
        self._run()
        recv = self.lm.last_call_kwargs["inputs_embeds"]
        image_mask = self.batch["image_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if not image_mask[b, t]:
                    self.assertTrue(
                        torch.allclose(recv[b, t], original[b, t], atol=1e-6),
                        f"Non-image position ({b},{t}) was modified",
                    )

    # --- attention mask ---

    def test_image_positions_excluded_from_attention_mask(self):
        self._run()
        recv_attn = self.lm.last_call_kwargs["attention_mask"]
        image_mask = self.batch["image_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if image_mask[b, t]:
                    self.assertEqual(
                        recv_attn[b, t].item(), 0,
                        f"Image position ({b},{t}) attention_mask should be 0",
                    )

    def test_text_positions_kept_in_attention_mask(self):
        self._run()
        recv_attn = self.lm.last_call_kwargs["attention_mask"]
        image_mask = self.batch["image_mask"]
        orig_attn = self.batch["attention_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if not image_mask[b, t] and orig_attn[b, t] == 1:
                    self.assertEqual(
                        recv_attn[b, t].item(), 1,
                        f"Text position ({b},{t}) attention_mask should be 1",
                    )

    # --- deepstack absent ---

    def test_deepstack_visual_embeds_not_passed_to_lm(self):
        self._run()
        self.assertNotIn(
            "deepstack_visual_embeds",
            self.lm.last_call_kwargs,
            "CachedText must NOT pass deepstack_visual_embeds to LM",
        )

    def test_visual_pos_masks_not_passed_to_lm(self):
        self._run()
        self.assertNotIn(
            "visual_pos_masks",
            self.lm.last_call_kwargs,
            "CachedText must NOT pass visual_pos_masks to LM",
        )

    # --- position ids ---

    def test_position_ids_recomputed_text_only(self):
        self._run()
        position_ids = self.batch["position_ids"]
        image_mask = self.batch["image_mask"].bool()
        is_text = (position_ids > 0) & (~image_mask)
        expected = torch.cumsum(is_text.long(), dim=-1) - 1
        expected = torch.where(is_text, expected, torch.zeros_like(expected))
        recv_pos = self.lm.last_call_kwargs["position_ids"]
        self.assertTrue(torch.equal(recv_pos, expected))

    # --- output structure ---

    def test_output_structure(self):
        out = self._run()
        self.assertEqual(out["preds"]["combined"].shape, (B, 2))
        self.assertEqual(out["features"]["combined"].shape, (B, D_MODEL))
        self.assertEqual(out["losses"], {})

    def test_ce_loss_when_label_provided(self):
        out = self._run(label=torch.tensor([0, 1]))
        self.assertIn("ce_loss_combined", out["losses"])
        self.assertEqual(out["losses"]["ce_loss_combined"].shape, ())

    # --- generation ---

    def test_generation_uses_masked_embeds(self):
        self._run(do_generate=True)
        gen_emb = self.model.backbone.generate_call_kwargs["inputs_embeds"]
        image_mask = self.batch["image_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if image_mask[b, t]:
                    self.assertAlmostEqual(
                        gen_emb[b, t, 0].item(), 1e-5, places=7,
                        msg=f"generate() embeds at image pos ({b},{t}) should be 1e-5",
                    )

    def test_generation_attention_excludes_image(self):
        self._run(do_generate=True)
        gen_attn = self.model.backbone.generate_call_kwargs["attention_mask"]
        image_mask = self.batch["image_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if image_mask[b, t]:
                    self.assertEqual(gen_attn[b, t].item(), 0)

    # --- edge cases ---

    def test_all_image_mask_no_crash(self):
        batch = _make_cached_batch()
        batch["image_mask"][:] = True
        batch["visual_pos_masks"][:] = True
        self._run(batch=batch)   # must not raise

    def test_no_image_mask_embeds_unchanged(self):
        batch = _make_cached_batch()
        batch["image_mask"][:] = False
        batch["visual_pos_masks"][:] = False
        original = batch["input_embeds"].clone()
        self._run(batch=batch)
        recv = self.lm.last_call_kwargs["inputs_embeds"]
        self.assertTrue(torch.allclose(recv, original, atol=1e-7))


# ---------------------------------------------------------------------------
# TestCachedImageMasking
# ---------------------------------------------------------------------------
class TestCachedImageMasking(unittest.TestCase):

    def setUp(self):
        self.model = _make_image_model()
        self.lm = self.model.backbone._lm
        self.batch = _make_cached_batch()

    def _run(self, batch=None, **kwargs):
        b = batch if batch is not None else self.batch
        return self.model.forward(b, **kwargs)

    # --- embedding masking ---

    def test_hint_positions_set_to_near_zero(self):
        self._run()
        recv = self.lm.last_call_kwargs["inputs_embeds"]
        hint_mask = self.batch["hint_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if hint_mask[b, t]:
                    self.assertTrue(
                        (recv[b, t] - 1e-5).abs().max() < 1e-7,
                        f"Hint position ({b},{t}) expected 1e-5",
                    )

    def test_non_hint_positions_unchanged(self):
        original = self.batch["input_embeds"].clone()
        self._run()
        recv = self.lm.last_call_kwargs["inputs_embeds"]
        hint_mask = self.batch["hint_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if not hint_mask[b, t]:
                    self.assertTrue(
                        torch.allclose(recv[b, t], original[b, t], atol=1e-6),
                        f"Non-hint position ({b},{t}) was modified",
                    )

    # --- attention mask ---

    def test_hint_positions_excluded_from_attention_mask(self):
        self._run()
        recv_attn = self.lm.last_call_kwargs["attention_mask"]
        hint_mask = self.batch["hint_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if hint_mask[b, t]:
                    self.assertEqual(recv_attn[b, t].item(), 0)

    def test_image_positions_kept_in_attention_mask(self):
        self._run()
        recv_attn = self.lm.last_call_kwargs["attention_mask"]
        image_mask = self.batch["image_mask"]
        hint_mask = self.batch["hint_mask"]
        orig_attn = self.batch["attention_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                # image positions are NOT hint positions → should remain attended
                if image_mask[b, t] and not hint_mask[b, t] and orig_attn[b, t] == 1:
                    self.assertEqual(recv_attn[b, t].item(), 1)

    # --- deepstack present ---

    def test_deepstack_visual_embeds_IS_passed_to_lm(self):
        self._run()
        self.assertIn(
            "deepstack_visual_embeds",
            self.lm.last_call_kwargs,
            "CachedImage MUST pass deepstack_visual_embeds to LM",
        )

    def test_visual_pos_masks_IS_passed_to_lm(self):
        self._run()
        self.assertIn(
            "visual_pos_masks",
            self.lm.last_call_kwargs,
            "CachedImage MUST pass visual_pos_masks to LM",
        )

    def test_visual_pos_masks_equals_image_mask(self):
        self._run()
        recv_vpm = self.lm.last_call_kwargs["visual_pos_masks"]
        self.assertTrue(torch.equal(recv_vpm.bool(), self.batch["image_mask"].bool()))

    # --- position ids ---

    def test_position_ids_recomputed_image_only(self):
        self._run()
        position_ids = self.batch["position_ids"]
        hint_mask = self.batch["hint_mask"].bool()
        is_text = (position_ids > 0) & (~hint_mask)
        expected = torch.where(
            is_text,
            torch.cumsum(is_text.long(), dim=-1) - 1,
            torch.zeros_like(position_ids),
        )
        self.assertTrue(torch.equal(self.lm.last_call_kwargs["position_ids"], expected))

    # --- output structure ---

    def test_output_structure(self):
        out = self._run()
        self.assertEqual(out["preds"]["combined"].shape, (B, 2))
        self.assertEqual(out["features"]["combined"].shape, (B, D_MODEL))
        self.assertEqual(out["losses"], {})

    def test_ce_loss_when_label_provided(self):
        out = self._run(label=torch.tensor([0, 1]))
        self.assertIn("ce_loss_combined", out["losses"])
        self.assertEqual(out["losses"]["ce_loss_combined"].shape, ())

    # --- generation ---

    def test_generation_uses_masked_embeds(self):
        self._run(do_generate=True)
        gen_emb = self.model.backbone.generate_call_kwargs["inputs_embeds"]
        hint_mask = self.batch["hint_mask"]
        for b in range(B):
            for t in range(T_SEQ):
                if hint_mask[b, t]:
                    self.assertAlmostEqual(gen_emb[b, t, 0].item(), 1e-5, places=7)

    # --- edge cases ---

    def test_all_hint_mask_no_crash(self):
        batch = _make_cached_batch()
        batch["hint_mask"][:] = True
        self._run(batch=batch)

    def test_no_hint_mask_embeds_unchanged(self):
        batch = _make_cached_batch()
        batch["hint_mask"][:] = False
        original = batch["input_embeds"].clone()
        self._run(batch=batch)
        recv = self.lm.last_call_kwargs["inputs_embeds"]
        self.assertTrue(torch.allclose(recv, original, atol=1e-7))


# ---------------------------------------------------------------------------
# TestCachedModalityComparison
# ---------------------------------------------------------------------------
class TestCachedModalityComparison(unittest.TestCase):

    def setUp(self):
        self.text_model = _make_text_model()
        self.image_model = _make_image_model()
        self.text_lm = self.text_model.backbone._lm
        self.image_lm = self.image_model.backbone._lm

    def test_text_vs_image_deepstack_difference(self):
        batch = _make_cached_batch()
        self.text_model.forward(batch)
        self.image_model.forward(batch)
        self.assertNotIn("deepstack_visual_embeds", self.text_lm.last_call_kwargs)
        self.assertIn("deepstack_visual_embeds", self.image_lm.last_call_kwargs)

    def test_text_model_output_invariant_to_vision_embeds(self):
        """Changing image-position embeddings must NOT affect CachedText output."""
        batch1 = _make_cached_batch()
        batch2 = copy.deepcopy(batch1)
        # Randomise image-position embeddings in batch2
        batch2["input_embeds"][:, 1:1 + N_IMG, :] = torch.rand(B, N_IMG, D_MODEL)

        out1 = self.text_model.forward(batch1)
        out2 = self.text_model.forward(batch2)
        self.assertTrue(
            torch.allclose(out1["preds"]["combined"], out2["preds"]["combined"]),
            "CachedText output should be invariant to image-position embeddings",
        )

    def test_image_model_output_invariant_to_hint_embeds(self):
        """Changing hint-position embeddings must NOT affect CachedImage output."""
        batch1 = _make_cached_batch()
        batch2 = copy.deepcopy(batch1)
        n_hint = 2  # matches _make_cached_batch default
        hint_start = 1 + N_IMG
        batch2["input_embeds"][:, hint_start:hint_start + n_hint, :] = torch.rand(B, n_hint, D_MODEL)

        out1 = self.image_model.forward(batch1)
        out2 = self.image_model.forward(batch2)
        self.assertTrue(
            torch.allclose(out1["preds"]["combined"], out2["preds"]["combined"]),
            "CachedImage output should be invariant to hint-position embeddings",
        )

    def test_text_model_output_varies_with_text_embeds(self):
        """CachedText output MUST change when the CLS-position embedding changes.

        CLS is at T-1 (a text / non-image position); it is directly extracted as h_cls.
        We flip its sign so layer_norm(h) → -layer_norm(h) → linear gives different logits.
        """
        batch1 = _make_cached_batch()
        batch3 = copy.deepcopy(batch1)
        # Flip CLS embedding direction — layer_norm(-h) = -layer_norm(h) → logits differ
        batch3["input_embeds"][:, T_SEQ - 1, :] *= -1.0

        out1 = self.text_model.forward(batch1)
        out3 = self.text_model.forward(batch3)
        self.assertFalse(
            torch.allclose(out1["preds"]["combined"], out3["preds"]["combined"]),
            "CachedText output should vary when CLS-position embedding direction changes",
        )

    def test_image_model_output_varies_with_image_embeds(self):
        """CachedImage output MUST change when deepstack_visual_embeds changes.

        CachedImage forwards deepstack to the LM (unlike CachedText).  _FakeLM adds
        dsv_mean into every hidden position; flipping deepstack flips that contribution
        and thus changes the direction of h_cls → different logits.
        """
        batch1 = _make_cached_batch()
        batch4 = copy.deepcopy(batch1)
        # Flip deepstack sign — changes dsv_mean direction → h_cls direction changes
        batch4["deepstack_visual_embeds"] = -batch4["deepstack_visual_embeds"]

        out1 = self.image_model.forward(batch1)
        out4 = self.image_model.forward(batch4)
        self.assertFalse(
            torch.allclose(out1["preds"]["combined"], out4["preds"]["combined"]),
            "CachedImage output should vary when deepstack_visual_embeds direction changes",
        )


if __name__ == "__main__":
    unittest.main()
