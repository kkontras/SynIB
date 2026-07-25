import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from synib.utils.configuration.config import process_config, setup_logger, process_config_default
from synib.training.pipeline import *

# xrandr --output DP-4 --scale 0.8x0.8

import argparse
import logging
import shutil
shutil._USE_CP_SENDFILE = False


def _inject_ironic_rate_in_template(template, ironic_rate):
    rate_token = "ir{}".format(float(ironic_rate))
    if "_ir" in template:
        return re.sub(r"_ir[0-9.]+", "_{}".format(rate_token), template, count=1)
    return template

def main(config_path, default_config_path, args):
    setup_logger()

    config = process_config_default(config_path, default_config_path)

    m = ""
    enc_m = ""

    if "fold" in args and args.fold is not None:
        if "data_split" in config.dataset:
            config.dataset.data_split.fold = int(args.fold)
        config.dataset.fold = int(args.fold)
        m += "fold{}".format(args.fold)
        enc_m += "{}".format(args.fold)
        seeds = ([0, 109, 19, 337, 42, 7, 1234, 5678, 999, 12345]
                 if "UCF" in config_path else
                 [109, 19, 337, 42, 7, 1234, 5678, 999, 12345, 54321])
        config.training_params.seed = int(seeds[int(args.fold)])
        if "norm_wav_path" in config.dataset:
            config.dataset.norm_wav_path = config.dataset.norm_wav_path.format(args.fold)
        if "norm_face_path" in config.dataset:
            config.dataset.norm_face_path = config.dataset.norm_face_path.format(args.fold)
        # if hasattr(config.model, "encoders"):
        #     for i in range(len(config.model.encoders)):
        #     # for i in range(2):
        #         config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(args.fold)
        # if "pretraining_paths" in config.model.args:
        #     for i in config.model.args.pretraining_paths:
        #         config.model.args.pretraining_paths[i] = config.model.args.pretraining_paths[i].format(args.fold)
    if "alpha" in args and args.alpha is not None:
        config.model.args.bias_infusion.alpha = float(args.alpha)
        m += "_alpha{}".format(args.alpha)
    if "recon_weight1" in args and args.recon_weight1 is not None:
        config.model.args.bias_infusion.weight1 = float(args.recon_weight1)
        m += "_w1{}".format(args.recon_weight1)
    if "recon_weight2" in args and args.recon_weight2 is not None:
        config.model.args.bias_infusion.weight2 = float(args.recon_weight2)
        m += "_w2{}".format(args.recon_weight2)
    if "recon_epochstages" in args and args.recon_epochstages is not None:
        config.model.args.bias_infusion.epoch_stages = int(args.recon_epochstages)
        m += "_epochstage{}".format(args.recon_epochstages)
    if "recon_ensemblestages" in args and args.recon_ensemblestages is not None:
        config.model.args.bias_infusion.ensemble_stages = int(args.recon_ensemblestages)
        m += "_ensstage{}".format(args.recon_ensemblestages)
    if "num_classes" in args and args.num_classes is not None:
        config.model.args.num_classes = int(args.num_classes)
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].args.num_classes = int(args.num_classes)
        # enc_m += "_numclasses{}".format(args.num_classes)
        m += "_numclasses{}".format(args.num_classes)
    if "tanh_mode_beta" in args and args.tanh_mode_beta is not None:
        config.model.args.bias_infusion.tanh_mode = "2"
        config.model.args.bias_infusion.tanh_mode_beta = float(args.tanh_mode_beta)
        m += "_beta{}".format(args.tanh_mode_beta)
    if "regby" in args and args.regby is not None:
        config.model.args.bias_infusion.regby = args.regby
        m += "_regby{}".format(args.regby)
    if "l" in args and args.l is not None:
        config.model.args.bias_infusion.l = float(args.l)
        m += "_l{}".format(args.l)
    if "multil" in args and args.multil is not None:
        for i in config.model.args.multi_loss.multi_supervised_w:
            if i != "combined" and config.model.args.multi_loss.multi_supervised_w[i] !=0:
                config.model.args.multi_loss.multi_supervised_w[i] = float(args.multil)
        m += "_multil{}".format(args.multil)
    if "lib" in args and args.lib is not None:
        config.model.args.bias_infusion.lib = float(args.lib)
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].args.lib = float(args.lib)
        m += "_lib{}".format(args.lib)
        # enc_m += "_lib{}".format(args.lib)
    if "kmepoch" in args and args.kmepoch is not None:
        config.model.args.bias_infusion.kmepoch = int(args.kmepoch)
        m += "_kmepoch{}".format(args.kmepoch)
    if "num_samples" in args and args.num_samples is not None:
        if "perturb" not in config.model.args:
            config.model.args.perturb = {}
        config.model.args.bias_infusion.num_samples = int(args.num_samples)
        config.model.args.perturb.num_samples = int(args.num_samples)
        m += "_numsamples{}".format(args.num_samples)

    if "contrcoeff" in args and args.contrcoeff is not None:
        config.model.args.bias_infusion.contr_coeff = float(args.contrcoeff)
        config.model.args.bias_infusion.contrcoeff = float(args.contrcoeff)
        m += "_contrcoeff{}".format(args.contrcoeff)

    if "shuffle_type" in args and args.shuffle_type is not None and args.shuffle_type != "None":
        config.model.args.bias_infusion.shuffle_type = str(args.shuffle_type)
        m += "_st{}".format(args.shuffle_type)

    if "validate_with" in args and args.validate_with is not None:
        config.early_stopping.validate_with = args.validate_with
        # enc_m += "_vld{}".format(args.validate_with)
        m += "_vld{}".format(args.validate_with)
    if "ironic_rate" in args and args.ironic_rate is not None:
        config.dataset.ironic_rate = float(args.ironic_rate)
        if hasattr(config.model, "ceu"):
            config.model.ceu.val = config.model.ceu.val.format("ir{}".format(float(args.ironic_rate)))
            config.model.ceu.test = config.model.ceu.test.format("ir{}".format(float(args.ironic_rate)))
        m += "_ir{}".format(float(args.ironic_rate))
    if "perturb" in args and args.perturb is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.type = args.perturb
        m += "_perturb{}".format(args.perturb)
    if "ending_epoch" in args and args.ending_epoch is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.ending_epoch = args.ending_epoch
        m += "_endingepoch{}".format(args.ending_epoch)
    if "perturb_fill" in args and args.perturb_fill is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.fill = args.perturb_fill
        m += "_fill{}".format(args.perturb_fill)
    if "perturb_pmin" in args and args.perturb_pmin is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.p_min = args.perturb_pmin
        m += "_pmin{}".format(args.perturb_pmin)
    if "perturb_lsparse" in args and args.perturb_lsparse is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.lsparse = args.perturb_lsparse
        m += "_lsparse{}".format(args.perturb_lsparse)
    if "perturb_pmax" in args and args.perturb_pmax is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.p_max = args.perturb_pmax
        m += "_pmax{}".format(args.perturb_pmax)
    if "debug_mask_stats" in args and args.debug_mask_stats:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.debug_mask_stats = True
    if "init_from" in args and args.init_from is not None:
        config.model.pretrained_model = {
            "use": True,
            "dir": args.init_from,
            "strict": bool(getattr(args, "init_strict", False)),
        }
    if getattr(args, "perturb_lsparse_sign", None) is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.lsparse_sign = args.perturb_lsparse_sign
        m += "_sign{}".format(args.perturb_lsparse_sign)
    if getattr(args, "l_anneal_epochs", None) is not None:
        if not hasattr(config.model.args, "bias_infusion"):
            config.model.args.bias_infusion = {}
        config.model.args.bias_infusion.l_anneal_epochs = int(args.l_anneal_epochs)
        m += "_lanneal{}".format(args.l_anneal_epochs)
    if getattr(args, "perturb_steps", None) is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.steps = int(args.perturb_steps)
        m += "_psteps{}".format(args.perturb_steps)
    if getattr(args, "perturb_lsparse_warmup", None) is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.lsparse_warmup = int(args.perturb_lsparse_warmup)
        m += "_lswarm{}".format(args.perturb_lsparse_warmup)
    if getattr(args, "perturb_lsparse_schedule", None) is not None:
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.lsparse_schedule = args.perturb_lsparse_schedule
        m += "_lssched{}".format(args.perturb_lsparse_schedule)
    if getattr(args, "persist_ell", False):
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.persist_ell = True
        m += "_persistell"
    if getattr(args, "save_masks", False):
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.save_masks = True
    if getattr(args, "penalty_swap", False):
        if not hasattr(config.model.args, "perturb"):
            config.model.args.perturb = {}
        config.model.args.perturb.penalty_swap = True
        m += "_penswap"
    if getattr(args, "disable_ce_random", False):
        config.model.args.synib_use_random_ce = False
        m += "_noCErand"
    if getattr(args, "disable_kl_learned", False):
        config.model.args.synib_use_learnable_kl = False
        m += "_noKLlearn"
    if getattr(args, "masking_only", False):
        config.model.args.synib_masking_only = True
        m += "_maskonly"
    # --- rebuttal reference ablation ---
    if getattr(args, "reference_type", None) is not None:
        config.model.args.reference_type = args.reference_type
        m += "_ref{}".format(args.reference_type)
    if getattr(args, "ref_diag", False):
        config.model.args.ref_diag = True
    if getattr(args, "ref_ema_decay", None) is not None:
        config.model.args.ref_ema_decay = float(args.ref_ema_decay)
        m += "_refema{}".format(args.ref_ema_decay)
    # per-fold anchor-init checkpoint selection: '{}' in ref_anchor_init paths is filled
    # from ref_anchor_seeds[fold] (HM: trained unimodal heads exist per seed, not per fold)
    if ("ref_anchor_init" in config.model.args and args.fold is not None
            and config.model.args.get("ref_anchor_seeds", None) is not None):
        _tok = str(config.model.args.ref_anchor_seeds[int(args.fold)])
        for _k in list(config.model.args.ref_anchor_init.keys()):
            config.model.args.ref_anchor_init[_k] = str(config.model.args.ref_anchor_init[_k]).format(_tok)
    if getattr(args, "modality_dropout", None) is not None and float(args.modality_dropout) > 0:
        config.model.args.modality_dropout = float(args.modality_dropout)
        m += "_mdrop{}".format(args.modality_dropout)
    if getattr(args, "modality_dropout_target", None) is not None:
        config.model.args.modality_dropout_target = args.modality_dropout_target
        m += "_tgt{}".format(args.modality_dropout_target)
    if getattr(args, "l_z1_masked", None) is not None:
        if not hasattr(config.model.args, "bias_infusion"):
            config.model.args.bias_infusion = {}
        config.model.args.bias_infusion.l_z1_masked = float(args.l_z1_masked)
        m += "_lz1m{}".format(args.l_z1_masked)
    if getattr(args, "l_z2_masked", None) is not None:
        if not hasattr(config.model.args, "bias_infusion"):
            config.model.args.bias_infusion = {}
        config.model.args.bias_infusion.l_z2_masked = float(args.l_z2_masked)
        m += "_lz2m{}".format(args.l_z2_masked)
    # --- clean (lambda, l_pareto) parametrisation ---
    # Only apply if user did NOT specify l_z1_masked / l_z2_masked explicitly
    if getattr(args, "l_pareto", None) is not None:
        if not hasattr(config.model.args, "bias_infusion"):
            config.model.args.bias_infusion = {}
        base_l = float(args.l) if (("l" in args) and (args.l is not None)) else \
                 float(getattr(config.model.args.bias_infusion, "l", 0.0))
        pareto = float(args.l_pareto)
        # Store l_pareto itself so model code can reference it by name.
        config.model.args.bias_infusion.l_pareto = pareto
        # l_z2_masked = lambda  (text branch stays at baseline)
        # l_z1_masked = lambda * l_pareto  (video branch scaled)
        if getattr(args, "l_z2_masked", None) is None:
            config.model.args.bias_infusion.l_z2_masked = base_l
        if getattr(args, "l_z1_masked", None) is None:
            config.model.args.bias_infusion.l_z1_masked = base_l * pareto
        m += "_pareto{}".format(args.l_pareto)
    if "optim_method" in args and args.optim_method is not None:
        config.model.args.bias_infusion.optim_method = args.optim_method
        m += "_optim{}".format(args.optim_method)
    if "lr" in args and args.lr is not None:
        config.optimizer.learning_rate = float(args.lr)
        m += "_lr{}".format(args.lr)
        # enc_m += "_lr{}".format(args.lr)
    if "wd" in args and args.wd is not None:
        config.optimizer.weight_decay = float(args.wd)
        m += "_wd{}".format(args.wd)
        # enc_m += "_wd{}".format(args.wd)
    if "cls" in args and args.cls is not None:
        config.model.args.cls_type = args.cls
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].args.cls_type = args.cls
        m += "_cls{}".format(args.cls)
    if "batch_size" in args and args.batch_size is not None:
        config.training_params.batch_size = int(args.batch_size)
        m += "_bs{}".format(args.batch_size)
        # enc_m += "_bs{}".format(args.batch_size)
    if "pre" in args and args.pre:
        m += "_pre"
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].pretrainedEncoder.use = True
    if "frozen" in args and args.frozen:
        m += "_frozen"
        print("Using frozen encoder")
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].args.freeze_encoder = True
    if "tdqm_disable" in args and args.tdqm_disable:
        config.training_params.tdqm_disable = True
    if "start_over" in args and args.start_over is not None:
        config.model.start_over = args.start_over
    if "no_model_save" in args and args.no_model_save:
        config.model.no_model_save = True

    if getattr(args, "num_layers", None) is not None:
        config.model.args.num_layers = int(args.num_layers)
        m += "_nlayers{}".format(args.num_layers)

    if getattr(args, "grad_accum", None) is not None:
        config.training_params.gradient_accumulation_steps = int(args.grad_accum)
        m += "_ga{}".format(args.grad_accum)

    if getattr(args, "tag", None) is not None:
        m = "{}_{}".format(args.tag, m)
    config.model.save_dir = config.model.save_dir.format(m)
    # expose save_dir to model args for mask-snapshot path (used by --save_masks)
    config.model.args.save_dir = config.model.save_dir

    if enc_m != "":
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                pre = config.model.encoders[i].get("pretrainedEncoder", None) if hasattr(config.model.encoders[i], "get") else getattr(config.model.encoders[i], "pretrainedEncoder", None)
                if pre is None or not getattr(pre, "dir", None):
                    continue
                config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(enc_m)
                if "ironic_rate" in args and args.ironic_rate is not None:
                    config.model.encoders[i].pretrainedEncoder.dir = _inject_ironic_rate_in_template(
                        config.model.encoders[i].pretrainedEncoder.dir,
                        args.ironic_rate,
                    )

    logging.info("save_dir: {}".format(config.model.save_dir))
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument('--config', help="Number of config file")
parser.add_argument('--default_config', help="Number of config file")
parser.add_argument('--fold', help="Fold")
parser.add_argument('--alpha', help="Alpha")
parser.add_argument('--tanh_mode_beta', help="tanh_mode_beta")
parser.add_argument('--regby', help="regby")
parser.add_argument('--batch_size', help="batch_size")
parser.add_argument('--l', help="L for Gat")
parser.add_argument('--multil', help="Coeff of Multi-Loss")
parser.add_argument('--lib', help="lib for Gat")
parser.add_argument('--kmepoch', help="keep memory epoch")
parser.add_argument('--num_samples', help="Number of samples for Gat")
parser.add_argument('--contrcoeff', help="ShuffleGrad Contrastive Coefficient")
parser.add_argument('--contr_type', help="ShuffleGrad Contrastive type")
parser.add_argument('--shuffle_type', help="shuffle_type")
parser.add_argument('--validate_with', help="validate_with")
parser.add_argument('--num_classes', help="num_classes")
parser.add_argument('--optim_method', help="Optim for Gat")
parser.add_argument('--ending_epoch', help="Ending epoch")
parser.add_argument('--load_ongoing', help="Ending epoch")
parser.add_argument('--recon_weight1', help="ReconBoost Parameters")
parser.add_argument('--recon_weight2', help="ReconBoost Parameters")
parser.add_argument('--recon_epochstages', help="ReconBoost Parameters")
parser.add_argument('--recon_ensemblestages', help="ReconBoost Parameters")
parser.add_argument('--lr', required=False, help="Learning Rate", default=None)
parser.add_argument('--wd', required=False, help="Weight Decay", default=None)
parser.add_argument('--cls', required=False, help="CLS linear, nonlinear, highlynonlinear", default=None)
parser.add_argument('--ironic_rate', required=False, help="Perturbation type of MCR", default=None)
parser.add_argument('--perturb', required=False, help="Perturbation type of MCR", default=None)
parser.add_argument('--perturb_fill', required=False, help="Fill for mask type perturbation of MCR", default=None)
parser.add_argument('--perturb_pmax', required=False, help="Fill for mask type perturbation of MCR", default=None)
parser.add_argument('--perturb_pmin', required=False, help="Fill for mask type perturbation of MCR", default=None)
parser.add_argument('--perturb_lsparse', required=False, help="Fill for mask type perturbation of MCR", default=None)
parser.add_argument('--rmask', required=False, help="Shortcut alias for --perturb and --perturb_fill", default=None)
parser.add_argument('--pmin', required=False, help="Shortcut alias for --perturb_pmin", default=None)
parser.add_argument('--pmax', required=False, help="Shortcut alias for --perturb_pmax", default=None)
parser.add_argument('--lsparse', required=False, help="Shortcut alias for --perturb_lsparse", default=None)
parser.add_argument('--debug_mask_stats', action='store_true',
                    help="Log learnable-mask inner-loop stats (g_mean, g_std, logit_norm, grad_norm, ce) to wandb + stdout for collapse diagnostics.")
parser.add_argument('--pre', action='store_true')
parser.add_argument('--frozen', action='store_true')
parser.add_argument('--tdqm_disable', action='store_true')
parser.add_argument('--start_over', action='store_true')
parser.add_argument('--no_model_save', action='store_true')
parser.add_argument('--num_layers', required=False, type=int, default=None,
                    help="Override num_layers in the model config (e.g. 1, 2, 3).")
parser.add_argument('--grad_accum', required=False, type=int, default=None,
                    help="Gradient accumulation steps (1 = no accumulation)")
parser.add_argument('--init_from', required=False, default=None,
                    help="Pretrained ckpt path (relative to save_base_dir) to initialize the model + first "
                         "encoder from. Loaded non-strict by default so SynIB-only params (synib.*) stay at init.")
parser.add_argument('--init_strict', action='store_true', default=False,
                    help="If set with --init_from, require strict state_dict match.")
parser.add_argument('--perturb_lsparse_sign', required=False, default=None,
                    choices=["keep", "destroy"],
                    help="Inner-loop sparsity term sign. 'destroy' (default, v1): lsparse*(1-g).mean(). "
                         "'keep': lsparse*g.mean() — encourages small sufficient keep-region.")
parser.add_argument('--l_anneal_epochs', required=False, type=int, default=None,
                    help="Linearly anneal outer synergy weight 0 → target over first N epochs. 0 = off.")
parser.add_argument('--perturb_steps', required=False, type=int, default=None,
                    help="Inner-loop Adam step count for the learned mask (default 5).")
parser.add_argument('--perturb_lsparse_warmup', required=False, type=int, default=None,
                    help="Number of initial inner steps with lsparse=0 (adversary warmup).")
parser.add_argument('--perturb_lsparse_schedule', required=False, default=None,
                    choices=["flat", "linear"],
                    help="How lsparse evolves across inner steps after warmup: "
                         "'flat' (default) or 'linear' ramp from 0 to lsparse.")
parser.add_argument('--persist_ell', action='store_true', default=False,
                    help="Warm-start the inner-loop mask logits `ell` from the previous batch "
                         "instead of re-initializing to ones. Tests the cold-start hypothesis.")
parser.add_argument('--save_masks', action='store_true', default=False,
                    help="Pickle g1_final, g2_final every ~200 steps under {save_dir}/masks/ for offline analysis.")
parser.add_argument('--penalty_swap', action='store_true', default=False,
                    help="Swap which branch carries which loss: CE on learned-mask counterfactuals, "
                         "KL on random-mask counterfactuals. Isolates mask-vs-loss in the 2x2 factorial.")
parser.add_argument('--disable_ce_random', action='store_true', default=False,
                    help="Set synib_use_random_ce=False. Skips the CE-on-random-mask branch entirely (F.10 cell).")
parser.add_argument('--disable_kl_learned', action='store_true', default=False,
                    help="Set synib_use_learnable_kl=False. Skips the learnable-mask branch entirely (F.01 cell).")
parser.add_argument('--masking_only', action='store_true', default=False,
                    help="Set synib_masking_only=True. Build masks and emit masked predictions, but skip ALL "
                         "KL terms; CE on the masked preds is then driven by multi_loss.multi_supervised_w. "
                         "Note: bias_infusion.l must remain > 0 (entry condition into the masked path).")
parser.add_argument('--modality_dropout', required=False, type=float, default=None,
                    help="Per-batch probability of zeroing one modality during training. "
                         "When hit, flips a coin between zeroing z1 or z2. Forces joint to learn "
                         "to recover under modality absence. Default 0.0 (off).")
parser.add_argument('--modality_dropout_target', required=False, default=None,
                    choices=["random", "z1", "z2"],
                    help="Which modality to target with modality_dropout. 'random' (default) flips a coin; "
                         "'z1' always zeros z1 (video); 'z2' always zeros z2 (text). "
                         "Use 'z2' to force the joint to rely on the WEAK modality (video).")
parser.add_argument('--l_z1_masked', required=False, type=float, default=None,
                    help="Outer KL weight specifically for the z1-masked (video destroyed) branch. "
                         "Overrides --l for kl_synergy_2. Up-weight to force joint to care when video is gone. "
                         "For the clean (lambda, l_pareto) parametrisation, use --l_pareto instead.")
parser.add_argument('--l_z2_masked', required=False, type=float, default=None,
                    help="Outer KL weight specifically for the z2-masked (text destroyed) branch. "
                         "Overrides --l for kl_synergy_1. Up-weight to force joint to care when text is gone. "
                         "For the clean (lambda, l_pareto) parametrisation, use --l_pareto instead.")
parser.add_argument('--l_pareto', required=False, type=float, default=None,
                    help="Asymmetry ratio between the two counterfactual KL branches. "
                         "Reparametrises the outer loss as  L_kl = l * (KL_text_branch + l_pareto * KL_video_branch). "
                         "l_pareto = 1 recovers the original symmetric SynIB/SynIB-U. "
                         "l_pareto > 1 up-weights the video-destroyed branch (use when video is the weak modality). "
                         "l_pareto < 1 up-weights the text-destroyed branch. "
                         "Sets l_z2_masked = l and l_z1_masked = l * l_pareto. "
                         "Ignored if --l_z1_masked or --l_z2_masked is given explicitly.")
parser.add_argument('--reference_type', required=False, default=None,
                    choices=["uniform", "class_prior", "unimodal_anchor", "anchor_legacy"],
                    help="Rebuttal reference ablation: reference distribution r in the masked-pred KL. "
                         "'uniform' = 1/K; 'class_prior' = fixed empirical train-label frequencies; "
                         "'unimodal_anchor' = EMA copy of the COMPLEMENTARY (unmasked) modality's unimodal "
                         "model (per App. H.5); 'anchor_legacy' = released-code direction (masked modality's "
                         "own clean prediction, live head). Default None = legacy synergy_type behavior.")
parser.add_argument('--ref_diag', action='store_true', default=False,
                    help="Log the reference-ablation diagnostic KL to frozen unimodal snapshots at every "
                         "validation pass (keys diag_kl_1/diag_kl_2 in val/test logs).")
parser.add_argument('--ref_ema_decay', required=False, type=float, default=None,
                    help="EMA decay for the unimodal_anchor reference copies (default 0.99).")
parser.add_argument('--tag', required=False, default=None,
                    help="Prefix tag added to save_dir and wandb run name for this sweep.")

parser.set_defaults(pre=False)
parser.set_defaults(start_over=False)
parser.set_defaults(no_model_save=False)
parser.set_defaults(frozen=False)
parser.set_defaults(tdqm_disable=False)
def cli():
    args = parser.parse_args()

    for var_name in vars(args):
        var_value = getattr(args, var_name)
        if var_value == "None":
            setattr(args, var_name, None)

    if args.rmask is not None:
        args.perturb = args.rmask
        if args.perturb_fill is None:
            args.perturb_fill = "ema"
    if args.pmin is not None:
        args.perturb_pmin = args.pmin
    if args.pmax is not None:
        args.perturb_pmax = args.pmax
    if args.lsparse is not None:
        args.perturb_lsparse = args.lsparse

    print(args)

    main(config_path=args.config, default_config_path=args.default_config, args=args)


if __name__ == "__main__":
    cli()
