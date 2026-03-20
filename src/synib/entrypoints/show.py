import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from colorama import Fore
from synib.utils.configuration.config import process_config, setup_logger, process_config_default
from synib.training.pipeline import *

# xrandr --output DP-4 --scale 0.8x0.8

import argparse
import logging
from synib.posthoc.Helpers.Helper_Importer import Importer
import numpy as np
import torch


def _to_1d_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().flatten().cpu()
    return torch.as_tensor(x).flatten().cpu()


def print_search(config_path, default_config_path, args):
    setup_logger()

    importer = Importer(config_name=config_path, default_files=default_config_path, device="cuda:0")
    m = ""
    if "fold" in args and args.fold is not None and args.fold != "None":
        m += "fold{}".format(args.fold)
    if "alpha" in args and args.alpha is not None and args.alpha != "None":
        m += "_alpha{}".format(args.alpha)
    if "recon_weight1" in args and args.recon_weight1 is not None:
        m += "_w1{}".format(args.recon_weight1)
    if "recon_weight2" in args and args.recon_weight2 is not None:
        m += "_w2{}".format(args.recon_weight2)
    if "recon_epochstages" in args and args.recon_epochstages is not None:
        m += "_epochstage{}".format(args.recon_epochstages)
    if "recon_ensemblestages" in args and args.recon_ensemblestages is not None:
        m += "_ensstage{}".format(args.recon_ensemblestages)
    if "num_classes" in args and args.num_classes is not None and args.num_classes != "None":
        m += "_numclasses{}".format(args.num_classes)
    if "tanh_mode_beta" in args and args.tanh_mode_beta is not None and args.tanh_mode_beta != "None":
        m += "_beta{}".format(args.tanh_mode_beta)
    if "regby" in args and args.regby is not None and args.regby != "None":
        m += "_regby{}".format(args.regby)
    if "l" in args and args.l is not None and args.l != "None":
        m += "_l{}".format(args.l)
    if "multil" in args and args.multil is not None:
        m += "_multil{}".format(args.multil)
    if "lib" in args and args.lib is not None and args.lib != "None":
        m += "_lib{}".format(args.lib)
    if "kmepoch" in args and args.kmepoch is not None and args.kmepoch != "None":
        m += "_kmepoch{}".format(args.kmepoch)
    if "ending_epoch" in args and args.ending_epoch is not None and args.ending_epoch != "None":
        m += "_endingepoch{}".format(args.ending_epoch)
    if "num_samples" in args and args.num_samples is not None and args.num_samples != "None":
        m += "_numsamples{}".format(args.num_samples)
    if "contrcoeff" in args and args.contrcoeff is not None and args.contrcoeff != "None":
        m += "_contrcoeff{}".format(args.contrcoeff)
    if "shuffle_type" in args and args.shuffle_type is not None and args.shuffle_type != "None":
        m += "_st{}".format(args.shuffle_type)
    if "contr_type" in args and args.contr_type is not None and args.contr_type != "None":
        m += "_contrtype{}".format(args.contr_type)
    if "validate_with" in args and args.validate_with is not None and args.validate_with != "None":
        m += "_vld{}".format(args.validate_with)

    if "ironic_rate" in args and args.ironic_rate is not None:
        m += "_ir{}".format(float(args.ironic_rate))

    if "perturb" in args and args.perturb is not None:
        m += "_perturb{}".format(args.perturb)
    if "ending_epoch" in args and args.ending_epoch is not None:
        m += "_endingepoch{}".format(args.ending_epoch)
    if "perturb_fill" in args and args.perturb_fill is not None:
        m += "_fill{}".format(args.perturb_fill)
    if "perturb_pmin" in args and args.perturb_pmin is not None:
        m += "_pmin{}".format(args.perturb_pmin)
    if "perturb_lsparse" in args and args.perturb_lsparse is not None:
        m += "_lsparse{}".format(args.perturb_lsparse)
    if "perturb_pmax" in args and args.perturb_pmax is not None:
        m += "_pmax{}".format(args.perturb_pmax)



    if "lr" in args and args.lr is not None:
        m += "_lr{}".format(args.lr)
    if "wd" in args and args.wd is not None:
        m += "_wd{}".format(args.wd)
    if "cls" in args and args.cls is not None:
        m += "_cls{}".format(args.cls)
    if "batch_size" in args and args.batch_size is not None:
        m += "_bs{}".format(args.batch_size)
    if "pre" in args and args.pre:
        m += "_pre"

    # ── IHA suffix (must match train.py injection order) ──
    if getattr(args, "pseudo_heads_q", None) is not None:
        m += "_phq{}".format(args.pseudo_heads_q)
    if getattr(args, "pseudo_heads_kv", None) is not None:
        m += "_phkv{}".format(args.pseudo_heads_kv)
    if getattr(args, "iha_init", None) is not None:
        m += "_ihainit{}".format(args.iha_init)
    if getattr(args, "iha_lr", None) is not None:
        m += "_ihalr{}".format(args.iha_lr)
    if getattr(args, "iha_layers", None) is not None:
        m += "_ihaL{}".format(args.iha_layers.replace(",", "-"))

    importer.config.model.save_dir = importer.config.model.save_dir.format(m)

    #Just to transfer the models
    # dst = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/CREMAD_models/MCR_models"
    # dst = os.path.join(dst,importer.config.model.save_dir)
    # import shutil
    # if "save_base_dir" in importer.config.model:
    #     importer.config.model.save_dir = os.path.join(importer.config.model.save_base_dir, importer.config.model.save_dir)
    # shutil.copyfile(importer.config.model.save_dir, dst)
    # print("Model transfered!")
    # return 0,0
    # if "save_base_dir" in importer.config.model:
    #     importer.config.model.save_dir = os.path.join(importer.config.model.save_base_dir, importer.config.model.save_dir)
    # os.remove(importer.config.model.save_dir)
    # print("Removed previous model: {}".format(importer.config.model.save_dir))
    # return 0, 0

    try:
        importer.load_checkpoint()
    except:
        # print("We could not load {}".format(config_path))
        print("We could not load {}".format(importer.config.model.save_dir))
        return 0, 0

    # print(importer.checkpoint["configs"])
    val_metrics, test_metric = importer.print_progress(multi_fold_results={},
                                                 verbose=False,
                                                 latex_version=False)


    message = Fore.WHITE + "{}  ".format(importer.config.model.save_dir.split("/")[-1])
    # val_metrics = multi_fold_results[0]
    # if "step" in val_metrics:
    #     message += Fore.GREEN + "Step: {}  ".format(val_metrics["step"])
    # if test_flag:
    #     message += Fore.RED + "Test  "
    if "current_epoch" in val_metrics:
        message += Fore.GREEN + "Epoch: {}  ".format(val_metrics["current_epoch"])
    if "steps_no_improve" in val_metrics:
        message += Fore.GREEN + "Steps no improve: {}  ".format(val_metrics["steps_no_improve"])
    # if "loss" in val_metrics:
    #     for i, v in val_metrics["loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i, v)
    if "acc" in val_metrics:
        for i, v in val_metrics["acc"].items():
            if i == "combined":
                message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.1f} ".format(i, v * 100)
    # print(test_metric)

    if test_metric and "acc" in test_metric:
        for i, v in test_metric["acc"].items():
            # if i == "combined":
                message += Fore.MAGENTA + "Test_Acc_{}: {:.1f} ".format(i, v * 100)

    if test_metric and "f1" in test_metric:
        for i, v in test_metric["f1"].items():
            # if i == "combined":
                message += Fore.MAGENTA + "Test_F1_{}: {:.1f} ".format(i, v * 100)

    if test_metric and "ceu" in val_metrics:
        message += Fore.LIGHTGREEN_EX + "V_CEU_{}: {:.3f} ".format("S", val_metrics["ceu"]["combined"]["synergy"])
        # for i, v in val_metrics["ceu"]["combined"].items(): message += Fore.LIGHTGREEN_EX + "V_CEU_{}: {:.2f} ".format(i, v)

    if test_metric and "ceu" in test_metric:
        message += Fore.LIGHTGREEN_EX + "T_CEU_{}: {:.3f} ".format("S", test_metric["ceu"]["combined"]["synergy"])
        # for i, v in test_metric["ceu"]["combined"].items(): message += Fore.LIGHTGREEN_EX + "T_CEU_{}: {:.2f} ".format(i, v)

    if test_metric and "f1_perclass" in test_metric:
        f1_vals = _to_1d_tensor(test_metric["f1_perclass"]["combined"]).tolist()
        rounded_v = ["{:.1f}".format(v * 100) for v in f1_vals]
        message += Fore.BLUE + "F1_perclass: {} ".format("-".join(rounded_v)) + Fore.RESET

    # if "top5_acc" in val_metrics:
    #     for i, v in val_metrics["top5_acc"].items(): message += Fore.LIGHTBLUE_EX + "Top5_Acc_{}: {:.2f} ".format(i,
    #                                                                                                               v * 100)
    # if "acc_exzero" in val_metrics:
    #     for i, v in val_metrics["acc_exzero"].items(): message += Fore.LIGHTBLUE_EX + "Acc_ExZ_{}: {:.2f} ".format(i,
    #                                                                                                                v * 100)
    # if "f1" in val_metrics:
    #     for i, v in val_metrics["f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, v * 100)
    # if "k" in val_metrics:
    #     for i, v in val_metrics["k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)
    # if "acc_7" in val_metrics:
    #     for i, v in val_metrics["acc_7"].items(): message += Fore.MAGENTA + "Acc7_{}: {:.4f} ".format(i, v * 100)
    # if "acc_5" in val_metrics:
    #     for i, v in val_metrics["acc_5"].items(): message += Fore.LIGHTMAGENTA_EX + "Acc5_{}: {:.4f} ".format(i, v * 100)
    # if "mae" in val_metrics:
    #     for i, v in val_metrics["mae"].items(): message += Fore.LIGHTBLUE_EX + "MAE_{}: {:.4f} ".format(i, v)
    # if "corr" in val_metrics:
    #     for i, v in val_metrics["corr"].items(): message += Fore.LIGHTWHITE_EX + "Corr_{}: {:.4f} ".format(i, v)
    # if "ece" in test_metric:
    #     for i, v in test_metric["ece"].items(): message += Fore.LIGHTWHITE_EX + "ECE_{}: {:.4f} ".format(i, v)
    if args.printing is True:
        print(message + Fore.RESET)
    return val_metrics, test_metric

from collections import defaultdict
def print_mean(m: dict, val=True):
    agg = {}
    counts = defaultdict(int)  # Keep track of counts for non-dict metrics

    # Step 1: Collect values
    for fold in m:
        for metric in m[fold]:
            if isinstance(m[fold][metric], dict):
                if metric not in agg:
                    agg[metric] = defaultdict(list)
                if metric == "f1_perclass":
                    agg[metric][pred].append(m[fold][metric][pred])
                for pred in m[fold][metric]:
                    # if pred == "combined":
                        agg[metric][pred].append(m[fold][metric][pred])
            else:
                if metric not in agg:
                    agg[metric] = []
                agg[metric].append(m[fold][metric])
                counts[metric] += 1

    # Step 2: Compute mean and std, and prepare the message
    message = ""
    if val:
        message += Fore.RED + "Val  "
    else:
        message += Fore.GREEN + "Test  "

    for metric in agg:
        if "acc" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.WHITE + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.1f} + {:.1f} ".format(100 * mean_value, 100 * std_value)
            else:
                mean_value = np.mean(agg[metric])
                std_value = np.std(agg[metric])
                message += Fore.WHITE + "{}: ".format(metric)
                message += Fore.LIGHTGREEN_EX + "{:.1f} + {:.1f} ".format(100 * mean_value, 100 * std_value)
        # if "acc_7" == metric:
        #     if isinstance(agg[metric], defaultdict):
        #         for pred in agg[metric]:
        #             mean_value = np.mean(agg[metric][pred])
        #             std_value = np.std(agg[metric][pred])
        #             message += Fore.GREEN + "{}_{}: ".format(metric, pred)
        #             message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        elif "f1" == metric:
            if isinstance(agg[metric], defaultdict):
                for pred in agg[metric]:
                    mean_value = np.mean(agg[metric][pred])
                    std_value = np.std(agg[metric][pred])
                    message += Fore.GREEN + "{}_{}: ".format(metric, pred)
                    message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        # elif "ece" == metric:
        #     if isinstance(agg[metric], defaultdict):
        #         for pred in agg[metric]:
        #             mean_value = np.mean(agg[metric][pred])
        #             std_value = np.std(agg[metric][pred])
        #             message += Fore.GREEN + "{}_{}: ".format(metric, pred)
        #             message += Fore.LIGHTGREEN_EX + "{:.4f} + {:.4f} ".format(mean_value, std_value)
        elif "synergy_gap_uni" in metric:
            message += Fore.LIGHTBLUE_EX + "SyG_Uni: {:.2f} ".format(metric["synergy_gap_uni"])
        elif "synergy_gap_ens" in metric:
            message += Fore.LIGHTBLUE_EX + "SyG_Ens: {:.2f} ".format(metric["synergy_gap_ens"])
        elif "ceu" == metric:
            pred = "combined"
            for each_ceu in agg[metric][pred][0]:
                mean_value = np.concatenate([np.array([i[each_ceu]]) for i in agg[metric][pred]]).mean()
                message += Fore.LIGHTBLUE_EX + "{}_{}: {:.2f} ".format(metric, each_ceu, mean_value)
        elif "f1_perclass" == metric:
            f1_perclass_vals = [_to_1d_tensor(i).unsqueeze(dim=0) for i in agg["f1_perclass"]["combined"]]
            mean_f1_irony_combined = torch.cat(f1_perclass_vals, dim=0).mean(dim=0)
            std_f1_irony_combined = torch.cat(f1_perclass_vals, dim=0).std(dim=0)
            message += Fore.LIGHTBLUE_EX + "{} ".format(metric)
            for i in range(len(mean_f1_irony_combined)):
                message += "{:.2f} ".format(mean_f1_irony_combined[i])




    if args.printing is True:
        print(message)

    mean_acc_combined = np.mean(agg["f1"]["combined"])
    std_acc_combined = np.std(agg["f1"]["combined"])

    mean_f1_irony_combined = _to_1d_tensor(mean_f1_irony_combined)[-1]
    std_f1_irony_combined = _to_1d_tensor(std_f1_irony_combined)[-1]

    return mean_acc_combined, std_acc_combined, mean_f1_irony_combined.item(), std_f1_irony_combined.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Command Line Program")
    parser.add_argument('--config', help="Number of config file")
    parser.add_argument('--default_config', help="Number of config file")
    parser.add_argument('--fold', help="Fold")
    parser.add_argument('--alpha', help="Alpha")
    parser.add_argument('--validate_with', help="validate_with")
    parser.add_argument('--tanh_mode_beta', help="tanh_mode_beta")
    parser.add_argument('--regby', help="regby")
    parser.add_argument('--batch_size', help="batch_size")
    parser.add_argument('--l', help="L for Gat")
    parser.add_argument('--multil', help="Coeff of Multi-Loss")
    parser.add_argument('--lib', help="L for Gat")
    parser.add_argument('--kmepoch', help="keep memory epoch")
    parser.add_argument('--num_samples', help="Number of samples for Gat")
    parser.add_argument('--contrcoeff', help="ShuffleGrad Contrastive Coefficient")
    parser.add_argument('--contr_type', help="ShuffleGrad Contrastive type")
    parser.add_argument('--shuffle_type', help="shuffle_type")
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
    parser.add_argument('--printing', required=False, help="print_results", default=True)
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
    parser.add_argument('--start_over', action='store_false')
    parser.add_argument('--pseudo_heads_q', required=False, type=int, default=None,
                        help="IHA pseudo-heads for Q")
    parser.add_argument('--pseudo_heads_kv', required=False, type=int, default=None,
                        help="IHA pseudo-heads for KV")
    parser.add_argument('--iha_init', required=False, default=None,
                        help="IHA init: identity, identity_noise, orthogonal")
    parser.add_argument('--iha_lr', required=False, type=float, default=None,
                        help="IHA mixing param learning rate")
    parser.add_argument('--iha_layers', required=False, default=None,
                        help="IHA layers: 'all' or comma-sep e.g. '20,21,22,23,24,25,26,27'")
    parser.set_defaults(pre=False)
    parser.set_defaults(start_over=False)
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

    config_li = list(args.config.split(","))
    val = {}
    test = {}
    if len(config_li) == 1:
        if "UCF" in args.config:
            for i in range(1,4):
                args.fold = i
                val_metric, test_metric = print_search(config_path=args.config, default_config_path=args.default_config, args=args)
                val[i] = val_metric
                test[i] = test_metric
        else:
            if args.fold is None:
                val_metric, test_metric = print_search(config_path=args.config, default_config_path=args.default_config,
                                                       args=args)
                val[0] = val_metric
                test[0] = test_metric
            else:
                for i in range(3):
                    args.fold = i
                    val_metric, test_metric = print_search(config_path=args.config, default_config_path=args.default_config, args=args)
                    val[i] = val_metric
                    test[i] = test_metric
    else:
        for i in config_li:
            val_metric, test_metric = print_search(config_path=i, default_config_path=args.default_config, args=args)
            val[i] = val_metric
            test[i] = test_metric

    try:
        mean_val, std_val, f1_irony_mean_val, f1_irony_std_val  = print_mean(val, val=True)
        mean_test, std_test, f1_irony_mean_test, f1_irony_std_test  = print_mean(test, val=False)
        # print(round(mean_val*100,1), round(std_val*100,1), round(mean_test*100,1), round(std_test*100,1))
        print(round(f1_irony_mean_test*100,1), round(f1_irony_std_test*100,1), "--", round(mean_test*100,1), round(std_test*100,1), "--")
        import sys
        sys.exit(mean_test, std_test)
    except:
        print("")
        # if args.printing is True:
        #     print("There was an error in the print_mean function")
        pass
