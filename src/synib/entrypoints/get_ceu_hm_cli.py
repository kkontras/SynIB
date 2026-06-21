"""
Extract CEU (Conditional Entropy of Union) unimodal predictions for Hateful Memes.

Mirrors the get_ceu_cli.py contract so the emitted pickles plug
straight into the SynIB trainer via config.model.ceu.{val,test}, but adapts to
HM conventions:
  - Seeds list instead of fold indices (HM runs use seeds 27/109/3407 directly).
  - Checkpoint filename pattern matches run_train_hm.sh / run_train_hm_save.sh:
        {save_base_dir}/HM_SmallTF_DeBERTa__{METHOD_TAG}_seed{SEED}.pth.tar
  - Two unimodal configs (text, image) in a single invocation; writes one
    val pickle and one test pickle with all 2×N_seeds entries folded in.

Example:
  PYTHONPATH=src python -m synib.entrypoints.get_ceu_hm_cli \
      --default_config run/configs/HatefulMemes/default_config_hm.json \
      --tier_config run/configs/HatefulMemes/tiers/small_tf_deberta.json \
      --unimodal_configs run/configs/HatefulMemes/methods/uni_text.json \
                         run/configs/HatefulMemes/methods/uni_image.json \
      --seeds 27 109 3407 \
      --output_root artifacts/ceus \
      --output_tag small_tf_deberta
"""
import argparse
import copy
import json
import os
import pickle
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from synib.posthoc.Helpers.Helper_Importer import Importer
from synib.utils.data.to_device import to_device

import torch
from collections import defaultdict
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
from scipy.stats import entropy as _entropy


def _eval_loader(model, loader, device):
    """Training-style eval: preserves dict-of-tensors modality entries."""
    model.eval()
    preds_list = []   # list of dict[head] -> logits
    labels_list = []
    with torch.no_grad():
        for served_dict in loader:
            if isinstance(served_dict, tuple):
                served_dict = {"data": {"c": served_dict[0][0],
                                        "f": served_dict[0][1],
                                        "g": served_dict[0][2]},
                               "label": served_dict[3].squeeze(dim=1)}
            data = to_device(served_dict["data"], device)
            label = to_device(served_dict["label"], device)
            out = model(data)
            pred = out["preds"]
            preds_list.append({k: v.detach().cpu() for k, v in pred.items()})
            labels_list.append(label.detach().cpu())
    # flatten
    keys = list(preds_list[0].keys())
    preds_cat = {k: torch.cat([p[k] for p in preds_list], dim=0).numpy() for k in keys}
    labels_cat = torch.cat(labels_list, dim=0).numpy()
    if labels_cat.ndim > 1:
        labels_cat = labels_cat.squeeze()
    return preds_cat, labels_cat


def _classification_metrics(pred_logits, labels):
    out = defaultdict(dict)
    for head, logits in pred_logits.items():
        probs = softmax(logits, axis=1)
        argmax = probs.argmax(axis=1)
        out["acc"][head] = float((argmax == labels).mean())
        out["f1"][head] = float(f1_score(labels, argmax, average="macro"))
        out["k"][head] = float(cohen_kappa_score(labels, argmax))
        try:
            if probs.shape[1] == 2:
                out["auroc"][head] = float(roc_auc_score(labels, probs[:, 1]))
            else:
                out["auroc"][head] = 0.0
        except Exception:
            out["auroc"][head] = 0.0
        out["conf"][head] = confusion_matrix(labels, argmax).tolist()
        out["entropy"][head] = float(_entropy(probs.T).mean())
    out["total_preds"] = {k: v for k, v in pred_logits.items()}
    out["total_preds_target"] = labels
    return dict(out)


def _to_serializable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64, np.float16)):
        return float(x)
    if isinstance(x, (np.int8, np.int16, np.int32, np.int64)):
        return int(x)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    return x


def _deep_merge(a: Dict, b: Dict) -> Dict:
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def _build_merged_config(default_config: str, tier_config: str, method_config: str,
                         seed: int, save_base_dir: str = None) -> str:
    """Mirror run_train_hm_save.sh: default ← tier ← method, set seed, and
    pre-resolve save_dir to match the on-disk filename produced by training."""
    cfg = json.load(open(default_config))
    tier = json.load(open(tier_config))
    method = json.load(open(method_config))
    _deep_merge(cfg, tier)
    _deep_merge(cfg, method)

    cfg["training_params"]["seed"] = int(seed)
    cfg.setdefault("dataset", {})
    cfg["dataset"].setdefault("fold", 0)
    cfg["dataset"].setdefault("data_split", {})
    cfg["dataset"]["data_split"].setdefault("fold", 0)

    method_tag = os.path.splitext(os.path.basename(method_config))[0]
    suffix = f"{method_tag}_seed{seed}"
    base = cfg["model"]["save_dir"].replace(".pth.tar", f"_{suffix}.pth.tar")
    # resolve the "{}" placeholder to empty string, matching the training run
    cfg["model"]["save_dir"] = base.format("")
    if save_base_dir:
        cfg["model"]["save_base_dir"] = save_base_dir
    cfg["model"]["start_over"] = False
    cfg["training_params"]["wandb_disable"] = True

    merged_path = tempfile.NamedTemporaryFile(
        prefix="ceu_hm_merged_", suffix=".json", delete=False).name
    with open(merged_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return merged_path


def _extract_fold_payload(results: Dict[str, Any], fold_key: int,
                          config_path: str, checkpoint: str,
                          seed: int, head: str = "combined") -> Dict[str, Any]:
    """Build a CEU fold payload. `head` selects which classifier head's logits
    go into the 'preds_combined' slot that the Evaluator reads. For HM unimodal
    checkpoints, the 'combined' head is untrained — pick the supervised
    unimodal head ('c' for uni_text, 'g' for uni_image)."""
    payload = {
        "fold": int(fold_key),
        "seed": int(seed),
        "config_path": config_path,
        "checkpoint": checkpoint,
        "head": head,
        "metrics": results,
        "preds_combined": None,
        "preds_all_heads": None,
        "targets": None,
    }
    try:
        payload["preds_combined"] = results["total_preds"][head]
    except Exception:
        pass
    try:
        payload["preds_all_heads"] = {k: results["total_preds"][k] for k in results["total_preds"]}
    except Exception:
        pass
    try:
        payload["targets"] = results["total_preds_target"]
    except Exception:
        pass
    return _to_serializable(payload)


def _evaluate(merged_config_path: str, default_config_path: str, device: str,
              test_batch_size: int) -> Dict[str, Any]:
    importer = Importer(config_name=merged_config_path,
                        default_files=default_config_path, device=device)
    importer.config.training_params.test_batch_size = int(test_batch_size)

    ckpt_full = os.path.join(
        importer.config.model.save_base_dir, importer.config.model.save_dir
    ) if importer.config.model.save_base_dir else importer.config.model.save_dir
    if not os.path.exists(ckpt_full):
        print(f"[WARN] missing checkpoint: {ckpt_full}")
        return {}

    importer.load_checkpoint()
    model = importer.get_model(return_model="best_model")
    data_loader = importer.get_dataloaders()
    model = model.to(device)

    val_preds, val_labels = _eval_loader(model, data_loader.valid_loader, device)
    test_preds, test_labels = _eval_loader(model, data_loader.test_loader, device)
    val_results = _classification_metrics(val_preds, val_labels)
    test_results = _classification_metrics(test_preds, test_labels)
    return {"checkpoint": ckpt_full, "val": val_results, "test": test_results}


def _load_existing(path: str, dataset: str, default_config_path: str,
                   unimodal_configs: List[str]) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {
        "dataset": dataset,
        "default_config_path": default_config_path,
        "unimodal_configs": list(unimodal_configs),
        "folds": {},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Extract HM unimodal CEU predictions.")
    p.add_argument("--default_config", required=True)
    p.add_argument("--tier_config", required=True)
    p.add_argument("--unimodal_configs", nargs="+", required=True,
                   help="One or more unimodal method configs (e.g. uni_text.json uni_image.json).")
    p.add_argument("--seeds", nargs="+", type=int, required=True,
                   help="Seed list; becomes the fold index ordering.")
    p.add_argument("--save_base_dir", default=None,
                   help="Override save_base_dir (where checkpoints live).")
    p.add_argument("--output_root", default="./artifacts/ceus")
    p.add_argument("--output_tag", default="",
                   help="Appended to output filename, e.g. small_tf_deberta.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--test_batch_size", type=int, default=64)
    p.add_argument("--allow_missing", action="store_true")
    p.add_argument("--dataset", default="hatefulmemes")
    args = p.parse_args()

    offset = len(args.seeds)
    output_dir = os.path.join(args.output_root, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    val_path = os.path.join(output_dir, f"{args.dataset}_ceu_val{tag}.pkl")
    test_path = os.path.join(output_dir, f"{args.dataset}_ceu_test{tag}.pkl")

    val_payload = _load_existing(val_path, args.dataset, args.default_config,
                                 args.unimodal_configs)
    test_payload = _load_existing(test_path, args.dataset, args.default_config,
                                  args.unimodal_configs)

    # Infer which head is supervised in each unimodal config (for HM:
    # uni_text.json sets c=1; uni_image.json sets g=1).
    def _supervised_head(path: str) -> str:
        try:
            m = json.load(open(path))
            w = m["model"]["args"]["multi_loss"]["multi_supervised_w"]
            winners = [k for k in ("c", "g", "combined") if w.get(k, 0) > 0]
            if winners:
                return winners[0]
        except Exception:
            pass
        # fallback by filename
        low = os.path.basename(path).lower()
        if "text" in low: return "c"
        if "image" in low or "vision" in low: return "g"
        return "combined"

    for mod_idx, method_config in enumerate(args.unimodal_configs):
        mod_offset = mod_idx * offset
        head = _supervised_head(method_config)
        print(f"[INFO] modality {mod_idx} ({os.path.basename(method_config)}) -> head='{head}'")
        for fold_idx, seed in enumerate(args.seeds):
            print(f"[INFO] {os.path.basename(method_config)}  seed={seed}  "
                  f"fold_key={fold_idx + mod_offset}")
            merged = _build_merged_config(
                args.default_config, args.tier_config, method_config,
                seed, save_base_dir=args.save_base_dir,
            )
            try:
                result = _evaluate(merged, args.default_config, args.device,
                                   args.test_batch_size)
            finally:
                try: os.remove(merged)
                except OSError: pass
            if not result:
                if args.allow_missing:
                    continue
                raise FileNotFoundError(f"Missing checkpoint for {method_config} seed {seed}")
            fk = int(fold_idx + mod_offset)
            val_payload["folds"][fk] = _extract_fold_payload(
                result["val"], fk, method_config, result["checkpoint"], seed, head=head)
            test_payload["folds"][fk] = _extract_fold_payload(
                result["test"], fk, method_config, result["checkpoint"], seed, head=head)

    with open(val_path, "wb") as f:
        pickle.dump(val_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(test_path, "wb") as f:
        pickle.dump(test_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[DONE] val  CEU -> {val_path}")
    print(f"[DONE] test CEU -> {test_path}")


if __name__ == "__main__":
    main()
