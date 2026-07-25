"""Unit tests for the rebuttal reference-ablation flag (model.args.reference_type).

Covers, per the experiment plan:
  - every reference is a valid distribution (rows sum to 1),
  - unimodal_anchor outputs change with the complementary input but carry no grad,
  - class_prior is fixed and normalized,
  - training losses are finite and backprop only into live parameters,
  - reference_type=None keeps the legacy loss keys (no semantic change to paper runs),
  - the EMA reference tracks the live encoder.

Run: PYTHONPATH=src python -m pytest tests/test_reference_types.py -q
"""

import copy

import pytest
import torch
import torch.nn as nn
from easydict import EasyDict

from synib.models.vlm.synib_mask_model import FusionIBModel_Mask, FusionIBModel_Mask_U

NUM_CLASSES = 3
D_MODEL = 8
B = 6


class DummyEncoder(nn.Module):
    """Minimal modality encoder honoring the SynIB encoder contract."""

    def __init__(self, key):
        super().__init__()
        self.key = key
        self.enc = nn.Linear(D_MODEL, D_MODEL)
        self.head = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward_uni(self, z, na_z=None, *, detach_pred=False, **kwargs):
        h = z.detach() if detach_pred else z
        return self.head(h)

    def forward(self, x, *, detach_pred=False, **kwargs):
        xi = x[self.key] if isinstance(x, dict) else x
        z = self.enc(xi)
        na_z = z.unsqueeze(1).repeat(1, 2, 1)
        return {
            "preds": {"combined": self.forward_uni(z, detach_pred=detach_pred)},
            "features": {"combined": z},
            "nonaggr_features": {"combined": na_z},
        }


def make_args(**over):
    args = EasyDict({
        "cls_type": "mlp",
        "num_classes": NUM_CLASSES,
        "d_model": D_MODEL,
        "fc_inner": 16,
        "in_dim": 16,
        "hidden_dim": 16,
        "dropout": 0.0,
        "norm_decision": False,
        "bias_infusion": {"l": 1.0},
        "multi_loss": {"multi_supervised_w": {"combined": 1, "c": 1, "g": 1}},
        "perturb": {"type": "rand", "p_min": 0.3, "fill": "ema", "noise_std": 1.0,
                    "num_samples": 1, "steps": 2, "lr": 0.1, "tau": 1.0, "lsparse": 1.0},
    })
    args.update(over)
    return args


def make_model(cls=FusionIBModel_Mask, **over):
    torch.manual_seed(0)
    encs = [DummyEncoder(0), DummyEncoder(1)]
    return cls(make_args(**over), encs)


def make_batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = {0: torch.randn(B, D_MODEL, generator=g), 1: torch.randn(B, D_MODEL, generator=g)}
    y = torch.randint(0, NUM_CLASSES, (B,), generator=g)
    return x, y


def _train_forward(model, x, y):
    model.train()
    return model(x, label=y)


# ---------------------------------------------------------------- validity

@pytest.mark.parametrize("ref", ["uniform", "class_prior", "unimodal_anchor", "anchor_legacy"])
def test_reference_is_valid_distribution(ref):
    over = {"reference_type": ref}
    if ref == "class_prior":
        over["class_prior"] = [4.0, 1.0, 1.0]
    model = make_model(**over)
    x, y = make_batch()
    out0 = model.enc_0(x)
    out1 = model.enc_1(x)
    r1, r2 = model._reference_probs(x, out0["preds"]["combined"], out1["preds"]["combined"])
    for r in (r1, r2):
        assert r.shape == (B, NUM_CLASSES)
        assert torch.allclose(r.sum(dim=-1), torch.ones(B), atol=1e-5)
        assert (r >= 0).all()
        assert not r.requires_grad


def test_class_prior_normalized_and_fixed():
    model = make_model(reference_type="class_prior", class_prior=[4.0, 1.0, 1.0])
    cp = model.ref_class_prior
    assert torch.allclose(cp, torch.tensor([4.0, 1.0, 1.0]) / 6.0)
    x, y = make_batch()
    _train_forward(model, x, y)
    assert torch.allclose(model.ref_class_prior, cp)  # unchanged by training


def test_class_prior_requires_prior():
    with pytest.raises(ValueError):
        make_model(reference_type="class_prior")


# ---------------------------------------------------------------- anchor semantics

def test_anchor_changes_with_complementary_input_and_carries_no_grad():
    model = make_model(reference_type="unimodal_anchor")
    x, y = make_batch()
    p1 = model.enc_0(x)["preds"]["combined"]
    p2 = model.enc_1(x)["preds"]["combined"]
    r1_a, r2_a = model._reference_probs(x, p1, p2)

    x_mod = {0: x[0] + 1.0, 1: x[1]}  # perturb modality 1 only
    r1_b, r2_b = model._reference_probs(x_mod, p1, p2)
    # branch _1 (z2 masked) is anchored on modality 1 → must move; branch _2 must not
    assert not torch.allclose(r1_a, r1_b)
    assert torch.allclose(r2_a, r2_b)
    assert not r1_a.requires_grad and r1_a.grad_fn is None


def test_anchor_direction_is_complementary_modality():
    """unimodal_anchor: branch _1 (z2 masked) ← EMA copy of enc_0 (modality 1)."""
    model = make_model(reference_type="unimodal_anchor")
    x, y = make_batch()
    r1, r2 = model._reference_probs(x, torch.zeros(B, NUM_CLASSES), torch.zeros(B, NUM_CLASSES))
    exp1 = torch.softmax(model.ref_enc_0(x, detach_pred=True)["preds"]["combined"], dim=-1)
    exp2 = torch.softmax(model.ref_enc_1(x, detach_pred=True)["preds"]["combined"], dim=-1)
    assert torch.allclose(r1, exp1, atol=1e-6)
    assert torch.allclose(r2, exp2, atol=1e-6)


def test_legacy_anchor_direction_preserved():
    """anchor_legacy must reproduce the released code: branch _1 target = uni_pred_2."""
    model = make_model(reference_type="anchor_legacy")
    x, _ = make_batch()
    p1 = torch.randn(B, NUM_CLASSES)
    p2 = torch.randn(B, NUM_CLASSES)
    r1, r2 = model._reference_probs(x, p1, p2)
    assert torch.allclose(r1, torch.softmax(p2, dim=-1))
    assert torch.allclose(r2, torch.softmax(p1, dim=-1))


# ---------------------------------------------------------------- training path

@pytest.mark.parametrize("ref", ["uniform", "class_prior", "unimodal_anchor", "anchor_legacy"])
def test_training_losses_finite_and_ref_gets_no_grad(ref):
    over = {"reference_type": ref}
    if ref == "class_prior":
        over["class_prior"] = [1.0, 2.0, 3.0]
    model = make_model(**over)
    x, y = make_batch()
    out = _train_forward(model, x, y)
    for k in ("kl_synergy_rand_1", "kl_synergy_rand_2", "kl_synergy_1", "kl_synergy_2"):
        assert k in out["losses"], k
        assert torch.isfinite(out["losses"][k]), k
    total = sum(out["losses"].values())
    total.backward()
    assert model.enc_2.net[0].weight.grad is not None  # fusion trunk trains
    if ref == "unimodal_anchor":
        for p in model.ref_enc_0.parameters():
            assert p.grad is None
        for p in model.ref_enc_1.parameters():
            assert p.grad is None


def test_legacy_path_unchanged_when_reference_type_none():
    model = make_model()  # synergy_type defaults to "gaussian", reference_type None
    assert model.reference_type is None
    assert not hasattr(model, "ref_enc_0")
    x, y = make_batch()
    out = _train_forward(model, x, y)
    for k in ("kl_synergy_rand_1", "kl_synergy_rand_2", "kl_synergy_1", "kl_synergy_2"):
        assert k in out["losses"]

    model_u = make_model(cls=FusionIBModel_Mask_U, synergy_type="unimodal_anchor")
    assert model_u.reference_type is None and model_u.synib.anchor_to_unimodal
    out_u = _train_forward(model_u, x, y)
    assert "kl_synergy_rand_1" in out_u["losses"]


def test_ema_update_tracks_live_encoder():
    model = make_model(reference_type="unimodal_anchor", ref_ema_decay=0.5)
    before = model.ref_enc_0.head.weight.clone()
    with torch.no_grad():
        model.enc_0.head.weight.add_(1.0)
    model._ref_ema_update()
    after = model.ref_enc_0.head.weight
    assert torch.allclose(after, before + 0.5, atol=1e-6)


# ---------------------------------------------------------------- diagnostics

def test_diag_kl_emitted_at_eval_only():
    model = make_model(reference_type="uniform", ref_diag=True)
    x, y = make_batch()
    out_tr = _train_forward(model, x, y)
    assert "diag" not in out_tr
    model.eval()
    with torch.no_grad():
        out_ev = model(x, label=y)
    assert "diag" in out_ev
    assert set(out_ev["diag"]) == {"diag_kl_1", "diag_kl_2"}
    for v in out_ev["diag"].values():
        assert torch.isfinite(v) and v >= 0
