import argparse
import os
import pickle
import sys
from typing import Any, Dict, List

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from synib.posthoc.Helpers.Helper_Importer import Importer
from synib.posthoc.Helpers.Helper_Validator import Validator


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


def _build_suffix(args: argparse.Namespace, fold: int) -> str:
    m = f"fold{fold}"
    if args.alpha is not None:
        m += f"_alpha{args.alpha}"
    if args.recon_weight1 is not None:
        m += f"_w1{args.recon_weight1}"
    if args.recon_weight2 is not None:
        m += f"_w2{args.recon_weight2}"
    if args.recon_epochstages is not None:
        m += f"_epochstage{args.recon_epochstages}"
    if args.recon_ensemblestages is not None:
        m += f"_ensstage{args.recon_ensemblestages}"
    if args.tanh_mode is not None:
        m += f"_tanhmode{args.tanh_mode}"
    if args.num_classes is not None:
        m += f"_numclasses{args.num_classes}"
    if args.tanh_mode_beta is not None:
        m += f"_beta{args.tanh_mode_beta}"
    if args.regby is not None:
        m += f"_regby{args.regby}"
    if args.clip is not None:
        m += f"_clip{args.clip}"
    if args.l is not None:
        m += f"_l{args.l}"
    if args.multil is not None:
        m += f"_multil{args.multil}"
    if args.l_diffsq is not None:
        m += f"_ldiffsq{args.l_diffsq}"
    if args.lib is not None:
        m += f"_lib{args.lib}"
    if args.ratio_us is not None:
        m += f"_ratio{args.ratio_us}"
    if args.kmepoch is not None:
        m += f"_kmepoch{args.kmepoch}"
    if args.mmcosine_scaling is not None:
        m += f"_mmcosinescaling{args.mmcosine_scaling}"
    if args.ilr_c is not None and args.ilr_g is not None:
        m += f"_ilrcg{args.ilr_c}_{args.ilr_g}"
    if args.ending_epoch is not None:
        m += f"_endingepoch{args.ending_epoch}"
    if args.num_samples is not None:
        m += f"_numsamples{args.num_samples}"
    if args.pow is not None:
        m += f"_pow{args.pow}"
    if args.nstep is not None:
        m += f"_nstep{args.nstep}"
    if args.contrcoeff is not None:
        m += f"_contrcoeff{args.contrcoeff}"
    if args.kde_coeff is not None:
        m += f"_kde_coeff{args.kde_coeff}"
    if args.etube is not None:
        m += f"_etube{args.etube}"
    if args.temperature is not None:
        m += f"_temp{args.temperature}"
    if args.shuffle_type is not None:
        m += f"_st{args.shuffle_type}"
    if args.contr_type is not None:
        m += f"_contrtype{args.contr_type}"
    if args.validate_with is not None:
        m += f"_vld{args.validate_with}"
    if args.base_alpha is not None:
        m += f"_basealpha{args.base_alpha}"
    if args.alpha_var is not None:
        m += f"_alphavar{args.alpha_var}"
    if args.base_beta is not None:
        m += f"_basebeta{args.base_beta}"
    if args.beta_var is not None:
        m += f"_betavar{args.beta_var}"
    if args.ironic_rate is not None:
        m += f"_ir{float(args.ironic_rate)}"
    if args.perturb is not None:
        m += f"_perturb{args.perturb}"
    if args.perturb_fill is not None:
        m += f"_fill{args.perturb_fill}"
    if args.perturb_pmin is not None:
        m += f"_pmin{args.perturb_pmin}"
    if args.perturb_lsparse is not None:
        m += f"_lsparse{args.perturb_lsparse}"
    if args.perturb_pmax is not None:
        m += f"_pmax{args.perturb_pmax}"
    if args.lr is not None:
        m += f"_lr{args.lr}"
    if args.wd is not None:
        m += f"_wd{args.wd}"
    if args.mm is not None:
        m += f"_mm{args.mm}"
    if args.cls is not None:
        m += f"_cls{args.cls}"
    if args.batch_size is not None:
        m += f"_bs{args.batch_size}"
    if args.pre:
        m += "_pre"
    return m


def _set_fold(importer: Importer, fold: int, config_path: str) -> None:
    if hasattr(importer.config, "dataset"):
        if "data_split" in importer.config.dataset:
            importer.config.dataset.data_split.fold = int(fold)
        importer.config.dataset.fold = int(fold)
        seeds = [0, 109, 19, 337] if "UCF" in config_path else [109, 19, 337]
        importer.config.training_params.seed = int(seeds[int(fold)])
        if "norm_wav_path" in importer.config.dataset:
            importer.config.dataset.norm_wav_path = importer.config.dataset.norm_wav_path.format(fold)
        if "norm_face_path" in importer.config.dataset:
            importer.config.dataset.norm_face_path = importer.config.dataset.norm_face_path.format(fold)


def _resolve_checkpoint_path(importer: Importer, suffix: str) -> str:
    importer.config.model.save_dir = importer.config.model.save_dir.format(suffix)
    if "save_base_dir" in importer.config.model:
        return os.path.join(importer.config.model.save_base_dir, importer.config.model.save_dir)
    return importer.config.model.save_dir


def _extract_fold_payload(results: Dict[str, Any], fold: int, config_path: str, checkpoint: str) -> Dict[str, Any]:
    payload = {
        "fold": int(fold),
        "config_path": config_path,
        "checkpoint": checkpoint,
        "metrics": results,
        "preds_combined": None,
        "targets": None,
    }
    try:
        payload["preds_combined"] = results["total_preds"]["combined"]
    except Exception:
        pass
    try:
        payload["targets"] = results["total_preds_target"]
    except Exception:
        pass
    return _to_serializable(payload)


def _evaluate(config_path: str, default_config_path: str, args: argparse.Namespace, fold: int) -> Dict[str, Any]:
    importer = Importer(config_name=config_path, default_files=default_config_path, device=args.device)
    _set_fold(importer, fold, config_path)
    importer.config.training_params.test_batch_size = int(args.test_batch_size)

    suffix = _build_suffix(args, fold)
    checkpoint = _resolve_checkpoint_path(importer, suffix)
    if not os.path.exists(checkpoint):
        if args.allow_missing:
            print(f"[WARN] Missing checkpoint, skipping: {checkpoint}")
            return {}
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    importer.load_checkpoint()
    model = importer.get_model(return_model="best_model")
    data_loader = importer.get_dataloaders()

    validator = Validator(model=model, data_loader=data_loader, config=importer.config, device=args.device)
    val_results = validator.get_results(set="Validation", print_results=args.print_results)
    test_results = validator.get_results(set="Test", print_results=args.print_results)
    return {
        "checkpoint": checkpoint,
        "val": val_results,
        "test": test_results,
    }


def _load_existing(path: str, dataset: str, default_config_path: str, unimodal_configs: List[str]) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {
        "dataset": dataset,
        "default_config_path": default_config_path,
        "unimodal_configs": list(unimodal_configs),
        "folds": {},
    }


def _sanitize_dataset_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CEU unimodal prediction pickles for val/test.")
    parser.add_argument("--dataset", required=True, help="Dataset label, e.g. ur_funny, mustard, cremad, scienceqa.")
    parser.add_argument("--default_config", required=True, help="Default config path.")
    parser.add_argument("--unimodal_configs", nargs=2, required=True, help="Two unimodal config paths.")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2], help="Fold list.")
    parser.add_argument("--output_root", default="./artifacts/ceus", help="Root output directory.")
    parser.add_argument("--output_tag", default="", help="Optional suffix tag in output filenames.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offset", type=int, default=None, help="Second unimodal fold index offset; default=len(folds).")
    parser.add_argument("--test_batch_size", type=int, default=6)
    parser.add_argument("--allow_missing", action="store_true", help="Skip missing checkpoints instead of failing.")
    parser.add_argument("--print_results", action="store_true")

    parser.add_argument("--alpha")
    parser.add_argument("--recon_weight1")
    parser.add_argument("--recon_weight2")
    parser.add_argument("--recon_epochstages")
    parser.add_argument("--recon_ensemblestages")
    parser.add_argument("--tanh_mode")
    parser.add_argument("--num_classes")
    parser.add_argument("--tanh_mode_beta")
    parser.add_argument("--regby")
    parser.add_argument("--clip")
    parser.add_argument("--l")
    parser.add_argument("--multil")
    parser.add_argument("--l_diffsq")
    parser.add_argument("--lib")
    parser.add_argument("--ratio_us")
    parser.add_argument("--kmepoch")
    parser.add_argument("--mmcosine_scaling")
    parser.add_argument("--ilr_c")
    parser.add_argument("--ilr_g")
    parser.add_argument("--ending_epoch")
    parser.add_argument("--num_samples")
    parser.add_argument("--pow")
    parser.add_argument("--nstep")
    parser.add_argument("--contrcoeff")
    parser.add_argument("--kde_coeff")
    parser.add_argument("--etube")
    parser.add_argument("--temperature")
    parser.add_argument("--shuffle_type")
    parser.add_argument("--contr_type")
    parser.add_argument("--validate_with")
    parser.add_argument("--base_alpha")
    parser.add_argument("--alpha_var")
    parser.add_argument("--base_beta")
    parser.add_argument("--beta_var")
    parser.add_argument("--ironic_rate")
    parser.add_argument("--perturb")
    parser.add_argument("--perturb_fill")
    parser.add_argument("--perturb_pmin")
    parser.add_argument("--perturb_lsparse")
    parser.add_argument("--perturb_pmax")
    parser.add_argument("--lr")
    parser.add_argument("--wd")
    parser.add_argument("--mm")
    parser.add_argument("--cls")
    parser.add_argument("--batch_size")
    parser.add_argument("--pre", action="store_true")

    args = parser.parse_args()

    for k, v in vars(args).items():
        if v == "None":
            setattr(args, k, None)

    offset = args.offset if args.offset is not None else len(args.folds)
    dataset_slug = _sanitize_dataset_name(args.dataset)
    output_dir = os.path.join(args.output_root, dataset_slug)
    os.makedirs(output_dir, exist_ok=True)

    tag = f"_{args.output_tag}" if args.output_tag else ""
    val_path = os.path.join(output_dir, f"{dataset_slug}_ceu_val{tag}.pkl")
    test_path = os.path.join(output_dir, f"{dataset_slug}_ceu_test{tag}.pkl")

    val_payload = _load_existing(val_path, dataset_slug, args.default_config, args.unimodal_configs)
    test_payload = _load_existing(test_path, dataset_slug, args.default_config, args.unimodal_configs)

    for mod_idx, unimodal_config in enumerate(args.unimodal_configs):
        mod_offset = mod_idx * offset
        for fold in args.folds:
            print(f"[INFO] Evaluating {unimodal_config} fold={fold} offset={mod_offset}")
            result = _evaluate(
                config_path=unimodal_config,
                default_config_path=args.default_config,
                args=args,
                fold=fold,
            )
            if not result:
                continue
            fold_key = int(fold + mod_offset)
            checkpoint = result["checkpoint"]
            val_payload["folds"][fold_key] = _extract_fold_payload(result["val"], fold_key, unimodal_config, checkpoint)
            test_payload["folds"][fold_key] = _extract_fold_payload(result["test"], fold_key, unimodal_config, checkpoint)

    with open(val_path, "wb") as f:
        pickle.dump(val_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(test_path, "wb") as f:
        pickle.dump(test_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[DONE] Saved val CEU to {val_path}")
    print(f"[DONE] Saved test CEU to {test_path}")


if __name__ == "__main__":
    main()
