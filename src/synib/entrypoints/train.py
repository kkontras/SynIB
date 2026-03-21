import os
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
        seeds = [0, 109, 19, 337] if "UCF" in config_path else [109, 19, 337]
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
        enc_m += "_ir{}".format(float(args.ironic_rate))
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

    # ── FullFT layer injection ──
    if getattr(args, "finetune_layers", None) is not None:
        if args.finetune_layers == "all":
            config.model.args.finetune_layers = "all"
        else:
            config.model.args.finetune_layers = [int(x) for x in args.finetune_layers.split(",")]
        m += "_ftL{}".format(args.finetune_layers.replace(",", "-"))

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

    config.model.save_dir = config.model.save_dir.format(m)

    if enc_m != "":
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(enc_m)

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
parser.add_argument('--pre', action='store_true')
parser.add_argument('--frozen', action='store_true')
parser.add_argument('--tdqm_disable', action='store_true')
parser.add_argument('--start_over', action='store_true')
parser.add_argument('--no_model_save', action='store_true')
parser.add_argument('--finetune_layers', required=False, default=None,
                    help="FullFT layers: 'all' or comma-sep e.g. '20,21,22,23,24,25,26,27'")
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

parser.set_defaults(pre=False)
parser.set_defaults(start_over=False)
parser.set_defaults(no_model_save=False)
parser.set_defaults(frozen=False)
parser.set_defaults(tdqm_disable=False)
args = parser.parse_args()

for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

if args.rmask is not None:
    args.perturb = args.rmask
    if args.perturb_fill is None:
        args.perturb_fill = args.rmask
if args.pmin is not None:
    args.perturb_pmin = args.pmin
if args.pmax is not None:
    args.perturb_pmax = args.pmax
if args.lsparse is not None:
    args.perturb_lsparse = args.lsparse

print(args)


main(config_path=args.config, default_config_path=args.default_config, args=args)
