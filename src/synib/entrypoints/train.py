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
    if "mmcosine_scaling" in args and args.mmcosine_scaling is not None:
        config.model.args.bias_infusion.mmcosine_scaling = float(args.mmcosine_scaling)
        m += "_mmcosinescaling{}".format(args.mmcosine_scaling)
    if "ilr_c" in args and "ilr_g" in args and args.ilr_c is not None and args.ilr_g is not None:
        config.model.args.bias_infusion.init_learning_rate = {
          "c" : float(args.ilr_c),
          "g" : float(args.ilr_g)
        }
        m += "_ilrcg{}_{}".format(args.ilr_c, args.ilr_g)
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
    if "base_alpha" in args and args.base_alpha is not None:
        config.dataset.base_alpha = float(args.base_alpha)
        m += "_basealpha{}".format(args.base_alpha)
    if "alpha_var" in args and args.alpha_var is not None:
        config.dataset.alpha_var = float(args.alpha_var)
        m += "_alphavar{}".format(args.alpha_var)
    if "base_beta" in args and args.base_beta is not None:
        config.dataset.base_beta = float(args.base_beta)
        config.model.args.layers = int(args.base_beta)
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                config.model.encoders[i].args.layers = int(args.base_beta)
        # enc_m += "_basebeta{}".format(args.base_beta)
        m += "_basebeta{}".format(args.base_beta)
    if "beta_var" in args and args.beta_var is not None:
        config.dataset.beta_var = float(args.beta_var)
        m += "_betavar{}".format(args.beta_var)
        if hasattr(config.model, "encoders"):
            for i in range(len(config.model.encoders)):
                m_enc = ""
                m_enc += "_basealpha{}".format(args.base_alpha)
                m_enc += "_alphavar{}".format(args.alpha_var)
                m_enc += "_basebeta{}".format(args.base_beta)
                m_enc += "_betavar{}".format(args.beta_var)
                config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(m_enc)
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
parser.add_argument('--tanh_mode', help="tanh_mode")
parser.add_argument('--tanh_mode_beta', help="tanh_mode_beta")
parser.add_argument('--regby', help="regby")
parser.add_argument('--clip', help="Gradient Clip Value")
parser.add_argument('--batch_size', help="batch_size")
parser.add_argument('--l', help="L for Gat")
parser.add_argument('--multil', help="Coeff of Multi-Loss")
parser.add_argument('--l_diffsq', help="L for Gat")
parser.add_argument('--lib', help="lib for Gat")
parser.add_argument('--ratio_us', help="lib for Gat")
parser.add_argument('--ratio_snr', help="lib for Gat")
parser.add_argument('--kmepoch', help="keep memory epoch")
parser.add_argument('--num_samples', help="Number of samples for Gat")
parser.add_argument('--pow', help="ShuffleGrad power")
parser.add_argument('--nstep', help="ShuffleGrad nstep Reg-Dist-Sep")
parser.add_argument('--contrcoeff', help="ShuffleGrad Contrastive Coefficient")
parser.add_argument('--kde_coeff', help="ShuffleGrad kde_coeff Coefficient")
parser.add_argument('--etube', help="ShuffleGrad Etube")
parser.add_argument('--temperature', help="ShuffleGrad Contrastive Temperature")
parser.add_argument('--contr_type', help="ShuffleGrad Contrastive type")
parser.add_argument('--shuffle_type', help="shuffle_type")
parser.add_argument('--validate_with', help="validate_with")
parser.add_argument('--transform_type', help="transform_type")
parser.add_argument('--trasform_before', help="trasform_before")
parser.add_argument('--num_classes', help="num_classes")
parser.add_argument('--base_alpha', help="Synthetic Alpha")
parser.add_argument('--alpha_var', help="Synthetic Alpha Variance")
parser.add_argument('--base_beta', help="Synthetic Beta")
parser.add_argument('--beta_var', help="Synthetic Beta Variance")
parser.add_argument('--optim_method', help="Optim for Gat")
parser.add_argument('--ilr_c', help="Initial Learning Rate Audio")
parser.add_argument('--ilr_g', help="Initial Learning Rate Video")
parser.add_argument('--mmcosine_scaling', help="mmcosine_scaling")
parser.add_argument('--ending_epoch', help="Ending epoch")
parser.add_argument('--load_ongoing', help="Ending epoch")
parser.add_argument('--commonlayers', help="Fusion with Conformer Layers")
parser.add_argument('--recon_weight1', help="ReconBoost Parameters")
parser.add_argument('--recon_weight2', help="ReconBoost Parameters")
parser.add_argument('--recon_epochstages', help="ReconBoost Parameters")
parser.add_argument('--recon_ensemblestages', help="ReconBoost Parameters")
parser.add_argument('--lr', required=False, help="Learning Rate", default=None)
parser.add_argument('--wd', required=False, help="Weight Decay", default=None)
parser.add_argument('--mm', required=False, help="Optimizer Momentum", default=None)
parser.add_argument('--cls', required=False, help="CLS linear, nonlinear, highlynonlinear", default=None)
parser.add_argument('--ironic_rate', required=False, help="Perturbation type of MCR", default=None)
parser.add_argument('--perturb', required=False, help="Perturbation type of MCR", default=None)
parser.add_argument('--perturb_fill', required=False, help="Fill for mask type perturbation of MCR", default=None)
parser.add_argument('--perturb_pmax', required=False, help="Fill for mask type perturbation of MCR", default=None)
parser.add_argument('--perturb_pmin', required=False, help="Fill for mask type perturbation of MCR", default=None)
parser.add_argument('--perturb_lsparse', required=False, help="Fill for mask type perturbation of MCR", default=None)
parser.add_argument('--pre', action='store_true')
parser.add_argument('--frozen', action='store_true')
parser.add_argument('--tdqm_disable', action='store_true')
parser.add_argument('--start_over', action='store_true')

parser.set_defaults(pre=False)
parser.set_defaults(start_over=False)
parser.set_defaults(frozen=False)
parser.set_defaults(tdqm_disable=False)
args = parser.parse_args()

for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

print(args)


main(config_path=args.config, default_config_path=args.default_config, args=args)
