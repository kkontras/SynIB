import torch
import logging
from torchmetrics import F1Score, CohenKappa, Accuracy
from collections import defaultdict
import torchmetrics
import numpy as np
from sklearn.metrics import f1_score
import pickle
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

from synib.utils.metrics.corr_metrics import (
    compare_feature_spaces,
    print_feature_comparisons,
    tsne_plot
)
import json

def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

VALIDATE_METHODS = ["accuracy", "loss", "syn_accuracy", "syn_loss"]

class All_Evaluator:
    def __init__(self, config, dataloaders: dict):

        task = config.get("task", "classification")

        if task == "classification" or task == "bias_measure":
            evaluator_class = General_Evaluator
        elif task == "regression":
            evaluator_class = General_Evaluator_Regression
        elif task == "generative":  # NEW
            evaluator_class = General_Evaluator_Generative
        else:
            raise ValueError(f"Unknown task type: {task}")

        self.train_evaluator = evaluator_class(config, len(dataloaders.train_loader.dataset), set="train")
        self.val_evaluator = evaluator_class(config, len(dataloaders.train_loader.dataset), set="val")
        if hasattr(dataloaders, "test_loader"):
            self.test_evaluator = evaluator_class(config, len(dataloaders.test_loader.dataset), set="test")

class General_Evaluator:
    def __init__(self, config, total_instances: int, set="val"):
        self.config = config
        self.set = set
        self.total_instances = total_instances
        self.num_classes = config.model.args.num_classes
        self.reset()

        self.logger = logging.getLogger("Evaluator")

        self.early_stop = False

        self.best_acc = 0.0
        self.best_loss = 0.0

        ceu = self.config.model.get("ceu",{})
        if set in ceu and ceu[set] is not None:
            with open(ceu[set], 'rb') as f:
                self.multi_fold_results_test = pickle.load(f)["folds"]
        else:
            logging.info("There is no CEU for {}".format(set))

    def set_best(self, best_acc, best_loss):
        self.best_acc = best_acc
        self.best_loss = best_loss
        self.logger\
            .info("Set current best acc {}, loss {}".format(self.best_acc, self.best_loss))

    def set_early_stop(self):
        self.early_stop = True

    def reset(self):
        self.losses = []
        self.preds = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.multi_supervised_w}
        self.features = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.multi_supervised_w}
        self.labels = []
        self.processed_instances = 0

    def process(self, all_output: dict):

        logits = {pred: all_output["pred"][pred].cpu() for pred in all_output["pred"]}
        if self.set == "val":
            features = {feat: all_output["features"][feat].cpu() for feat in all_output["features"]}
        label = all_output["label"].cpu()
        loss = {l_i: all_output["loss"][l_i].detach().cpu() for l_i in all_output["loss"]}
        num_instances = label.shape[0]

        for pred_key in logits:
            if pred_key not in self.preds:
                continue
            assert (len(logits[pred_key].shape) == 2), "The shape of logits must be in format [bs, num_test_clips * num_test_crops, total_classes]"
            self.preds[pred_key].append(logits[pred_key])

        if self.set == "val":
            for feat_key in features:
                if feat_key not in self.features:
                    continue
                self.features[feat_key].append(features[feat_key])

        self.labels.append(label)

        self.processed_instances += num_instances
        self.losses.append(loss)

    def get_early_stop(self):
        return self.early_stop

    def enable_early_stop(self):
        self.early_stop = True

    def mean_batch_loss(self):
        if len(self.losses)==0:
            return None, ""
        mean_batch_loss = {}
        for key in self.losses[0].keys():
            mean_batch_loss[key] = torch.stack([self.losses[i][key] for i in range(len(self.losses)) if key in self.losses[i]]).mean().item()

        message = ""
        for mean_key in mean_batch_loss: message += "{}: {:.3f} ".format(mean_key, mean_batch_loss[mean_key])

        return dict(mean_batch_loss), message

    def evaluate(self, **eval_kwargs) -> Dict[str, Any]:

        metrics  = defaultdict(dict)
        if len(self.labels) == 0: return {}
        targets = torch.concatenate(self.labels).cpu().flatten()
        mean_batch_loss, _ = self.mean_batch_loss()
        if mean_batch_loss is not None:
            metrics["loss"] = mean_batch_loss

        is_binary = len(torch.unique(targets)) == 2
        task_type = "binary" if is_binary else "multiclass"

        if is_binary:
            ece = torchmetrics.CalibrationError(num_classes=self.config.model.args.num_classes, task="BINARY")
        else:
            ece = torchmetrics.CalibrationError(num_classes=self.config.model.args.num_classes, task="MULTICLASS")

        def compute_metrics_for_head(pred_key: str, preds: torch.Tensor):
            head_metrics = {}

            if is_binary:
                targets_onehot = F.one_hot(targets.to(torch.int64), num_classes=self.num_classes).float()
                head_metrics["acc"] = Accuracy(task="binary", num_classes=self.num_classes)(preds,
                                                                                            targets_onehot).item()
                head_metrics["f1"] = F1Score(task="binary", num_classes=self.num_classes, average="macro")(preds,
                                                                                                           targets_onehot).item()
                head_metrics["f1_mi"] = F1Score(task="binary", num_classes=self.num_classes, average="micro")(preds,
                                                                                                              targets_onehot).item()
                head_metrics["k"] = CohenKappa(task="binary", num_classes=self.num_classes)(preds,
                                                                                            targets_onehot).item()
                head_metrics["f1_perclass"] = F1Score(task="binary", num_classes=self.num_classes, average=None)(preds,
                                                                                                                 targets_onehot)
                head_metrics["ece"] = ece(preds, targets_onehot).item()

            else:
                head_metrics["acc"] = Accuracy(task="multiclass", num_classes=self.num_classes)(preds, targets).item()
                head_metrics["f1"] = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")(preds,
                                                                                                               targets).item()
                head_metrics["f1_mi"] = F1Score(task="multiclass", num_classes=self.num_classes, average="micro")(preds,
                                                                                                                  targets).item()
                head_metrics["k"] = CohenKappa(task="multiclass", num_classes=self.num_classes)(preds, targets).item()
                head_metrics["f1_perclass"] = F1Score(task="multiclass", num_classes=self.num_classes, average=None)(
                    preds, targets)
                head_metrics["ece"] = ece(preds, targets).item()

                if self.num_classes > 5:
                    head_metrics["top5_acc"] = Accuracy(
                        task="multiclass", num_classes=self.num_classes, top_k=5
                    )(preds, targets).item()

            if pred_key == "combined" and (self.set == "val" or self.set == "test"):
                ceu_val = self.ceu(preds, targets)
                per_group_acc = self.get_per_group_accuracy(preds, targets)
                if ceu_val is not None:
                    head_metrics["ceu"] = ceu_val
                if per_group_acc is not None:
                    head_metrics["pg_acc"]=per_group_acc

            return head_metrics

        for pred_key, preds_list in self.preds.items():
            if len(preds_list) == 0:
                continue
            preds = torch.concatenate(preds_list).cpu()

            head_metrics = compute_metrics_for_head(pred_key, preds)
            for m_key, m_val in head_metrics.items():
                metrics.setdefault(m_key, {})[pred_key] = m_val

        synergy_keys = ["a", "b", "c", "combined"]
        if all(k in metrics["acc"] for k in synergy_keys):
            acc_combined = metrics["acc"]["combined"]
            acc_unimodal = {k: metrics["acc"][k] for k in ["a", "b", "c"]}
            best_uni_acc = max(acc_unimodal.values())

            preds_a = torch.concatenate(self.preds["a"]).cpu()
            preds_b = torch.concatenate(self.preds["b"]).cpu()
            preds_c = torch.concatenate(self.preds["c"]).cpu()
            preds_ensemble = (preds_a + preds_b + preds_c) / 3.0

            acc_ensemble = Accuracy(task=task_type, num_classes=self.num_classes)(
                preds_ensemble, targets if not is_binary else F.one_hot(targets, self.num_classes).float()
            ).item()

            metrics["acc"]["best_unimodal"] = best_uni_acc
            metrics["acc"]["ensemble_soft"] = acc_ensemble
            metrics["synergy_gap_uni"] = acc_combined - best_uni_acc
            metrics["synergy_gap_ens"] = acc_combined - acc_ensemble

        return metrics

    def ceu(self, total_preds, targets_tens):
        def create_conf(predictions):

            predictions = np.array(predictions)
            all_false = np.all(predictions[:2] == 0, axis=0)
            only_mod0_true = (predictions[0] == 1) & (predictions[1] == 0)
            only_mod1_true = (predictions[1] == 1) & (predictions[0] == 0)
            both_mods_true = (predictions[1] == 1) & (predictions[0] == 1)
            mmodel_true = predictions[2] == 1

            cm = np.array([
                [(~mmodel_true[all_false]).sum(), mmodel_true[all_false].sum()],
                [(~mmodel_true[only_mod0_true]).sum(), mmodel_true[only_mod0_true].sum()],
                [(~mmodel_true[only_mod1_true]).sum(), mmodel_true[only_mod1_true].sum()],
                [(~mmodel_true[both_mods_true]).sum(), mmodel_true[both_mods_true].sum()],
            ])
            mmodel_true[both_mods_true].sum()
            cm = 100 * cm.astype('float') / cm.sum()  # Normalize by row
            return cm

        this_fold = self.config.dataset.get("fold", 0)
        if not hasattr(self, "multi_fold_results_test"):
            logging.info("Model doesnt have CEU for split: ".format(self.set))
            return None

        audio_preds = np.array(self.multi_fold_results_test[this_fold]["preds_combined"])
        audio_targets = np.array(self.multi_fold_results_test[this_fold]["targets"])
        video_preds = np.array(self.multi_fold_results_test[this_fold+3]["preds_combined"])
        video_targets = np.array(self.multi_fold_results_test[this_fold+3]["targets"])

        if len(targets_tens) == len(video_targets) == len(audio_targets) and (targets_tens.numpy() == video_targets).all() and (video_targets == audio_targets).all():

            predictions = [ audio_preds.argmax(-1) == audio_targets,
                            video_preds.argmax(-1) == video_targets,
                            (total_preds.argmax(-1) == targets_tens).numpy(),]
            cm = create_conf(predictions)
            cm = np.round(cm, 2)
            cue_audio = cm[1, 1] / (cm[1].sum())
            cue_video = cm[2, 1] / (cm[2].sum())
            synergy = cm[0, 1] / (cm[0].sum())
            coexistence = cm[3, 1] / (cm[3].sum())
            return {"cue_audio": cue_audio, "cue_video": cue_video, "synergy":synergy, "coexistence":coexistence}
        else:
            logging.info(f"We got targets_tens: {targets_tens.shape} video_targets: {video_targets.shape} audio_targets: {audio_targets.shape}")
            raise ValueError(f"We got targets_tens: {targets_tens.shape} video_targets: {video_targets.shape} audio_targets: {audio_targets.shape} and they are equal by pairs T-V:{(targets_tens.numpy() == video_targets).all()} V-A:{(video_targets == audio_targets).all()}")
    def get_per_group_accuracy(self, total_preds, targets_tens):
        def calculate_stats(preds_correct, targets, m_preds):
            preds_correct = np.array(preds_correct)
            targets = np.array(targets)
            m_preds = np.array(m_preds)
            total_samples = float(len(targets))

            groups = {
                "synergy": (~preds_correct[0]) & (~preds_correct[1]),
                "cue_audio": (preds_correct[0]) & (~preds_correct[1]),
                "cue_video": (~preds_correct[0]) & (preds_correct[1]),
                "coexistence": (preds_correct[0]) & (preds_correct[1])
            }

            m_model_correct = preds_correct[2]
            results = {"group_metrics": {}}

            for name, mask in groups.items():
                group_size = int(mask.sum())
                correct_in_group = int(m_model_correct[mask].sum())

                internal_acc = (correct_in_group / group_size * 100.0) if group_size > 0 else 0.0
                contribution = (correct_in_group / total_samples * 100.0)

                results["group_metrics"][name] = {
                    "internal_acc": float(internal_acc),
                    "contribution_to_total": float(contribution)
                }

            syn_mask = groups["synergy"]
            unique_labels = np.unique(targets)
            if syn_mask.any():
                syn_targets = targets[syn_mask]
                syn_predictions = m_preds[syn_mask]
                f1_list = []
                for l in unique_labels:
                    tp = ((syn_predictions == l) & (syn_targets == l)).sum()
                    fp = ((syn_predictions == l) & (syn_targets != l)).sum()
                    fn = ((syn_predictions != l) & (syn_targets == l)).sum()
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_list.append(2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0)
                results["synergy_f1"] = float(np.mean(f1_list) * 100.0)
            else:
                results["synergy_f1"] = 0.0

            label_results = {}
            for label in unique_labels:
                l_mask = (targets == label) & syn_mask
                l_count = int(l_mask.sum())
                l_acc = (m_model_correct[l_mask].sum() / l_count * 100.0) if l_count > 0 else 0.0
                label_results[f"label_{int(label)}"] = float(l_acc)

            results["synergy_per_label_acc"] = label_results
            return results

        this_fold = self.config.dataset.get("fold", 0)
        if not hasattr(self, "multi_fold_results_test"): return None

        audio_preds = np.array(self.multi_fold_results_test[this_fold]["preds_combined"])
        audio_targets = np.array(self.multi_fold_results_test[this_fold]["targets"])
        video_preds = np.array(self.multi_fold_results_test[this_fold+3]["preds_combined"])
        video_targets = np.array(self.multi_fold_results_test[this_fold+3]["targets"])

        targets_np = targets_tens.numpy()
        if len(targets_np) == len(video_targets) == len(audio_targets) and \
                (targets_np == video_targets).all() and (video_targets == audio_targets).all():
            preds_correct = [
                audio_preds.argmax(-1) == audio_targets,
                video_preds.argmax(-1) == video_targets,
                (total_preds.argmax(-1) == targets_tens).numpy()
            ]
            return calculate_stats(preds_correct, targets_np, total_preds.argmax(-1).numpy())
        return None

    def is_best(self, metrics = None, best_logs=None):
        if metrics is None:
            metrics = self.evaluate()

        is_best_dict = {}
        if "loss" in metrics and "ce_loss_combined" in metrics["loss"]:
            is_best_dict["loss"] = metrics["loss"]["ce_loss_combined"] < best_logs["best_vloss"]["loss"]["ce_loss_combined"]
        else:
            is_best_dict["loss"] = False

        if "acc" in metrics and "combined" in metrics["acc"]:
            is_best_dict["accuracy"] = metrics["acc"]["combined"] > best_logs["best_vaccuracy"]["acc"]["combined"]
        else:
            is_best_dict["accuracy"] = False

        if "combined" in metrics["pg_acc"]:
            prev = best_logs["best_vsyn_accuracy"]["pg_acc"]["combined"]["group_metrics"]["synergy"]["internal_acc"]
            is_best_dict["syn_accuracy"] = metrics["pg_acc"]["combined"]["group_metrics"]["synergy"]["internal_acc"] > prev
        else:
            is_best_dict["syn_accuracy"] = False

        message = ""
        for key, value in is_best_dict.items():
            if value:
                message+= key + "-"
        if message != "":
            self.logger.info(f"New best with {message}")

        flag = any(is_best_dict.values())
        return flag, is_best_dict

class General_Evaluator_Regression:
    def __init__(self, config, total_instances: int, set="val"):
        self.config = config
        self.total_instances = total_instances
        self.num_classes = config.model.args.num_classes
        self.set = set
        self.reset()

        self.early_stop = False

        self.best_acc = 0.0
        self.best_loss = 0.0

    def set_best(self, best_acc, best_loss):
        self.best_acc = best_acc
        self.best_loss = best_loss
        self.logger.info("Set current best acc {}, loss {}".format(self.best_acc, self.best_loss))

    def reset(self):
        self.losses = []
        self.preds = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.multi_supervised_w if self.config.model.args.multi_loss.multi_supervised_w[pred_key] != 0.0}
        self.labels = []
        self.processed_instances = 0

    def set_early_stop(self):
        self.early_stop = True

    def process(self, all_output: dict):

        logits = all_output["pred"]
        label = all_output["label"]
        loss = all_output["loss"]
        num_instances = label.shape[0]

        for pred_key in logits:
            if pred_key not in self.preds:
                continue
            assert (len(logits[pred_key].shape) == 2), "The shape of logits must be in format [bs, num_test_clips * num_test_crops, total_classes]"
            self.preds[pred_key].append(logits[pred_key])

        self.labels.append(label)

        self.processed_instances += num_instances
        self.losses.append(loss)

    def get_early_stop(self):
        return self.early_stop

    def enable_early_stop(self):
        self.early_stop = True

    def mean_batch_loss(self):
        if len(self.losses)==0:
            return None, ""
        mean_batch_loss = {}
        for key in self.losses[0].keys():
            mean_batch_loss[key] = torch.stack([self.losses[i][key] for i in range(len(self.losses))]).mean().item()

        message = ""
        for mean_key in mean_batch_loss: message += "{}: {:.3f} ".format(mean_key, mean_batch_loss[mean_key])

        return dict(mean_batch_loss), message

    def evaluate(self, **eval_kwargs) -> Dict[str, Any]:

        targets_tens = torch.concatenate(self.labels).cpu().flatten()

        mean_batch_loss, _ = self.mean_batch_loss()

        total_preds, metrics  = {}, defaultdict(dict)
        if mean_batch_loss is not None:
            metrics["loss"] = mean_batch_loss

        ece = torchmetrics.CalibrationError(num_classes=self.config.model.args.num_classes, task="BINARY")
        for pred_key in self.preds:
            if len(self.preds[pred_key]) == 0:
                continue
            total_preds = torch.concatenate(self.preds[pred_key]).cpu().squeeze()#[:self.processed_instances]

            binary_truth_nozeros = (targets_tens[targets_tens!=0] > 0)
            binary_preds_nozeros = (total_preds[targets_tens!=0] > 0)

            binary_truth = (targets_tens > 0)
            binary_preds = (total_preds > 0)

            metrics["acc"][pred_key] = Accuracy(task="binary")(binary_preds_nozeros,binary_truth_nozeros).item()
            metrics["acc_has0"][pred_key] = Accuracy(task="binary")(binary_preds,binary_truth).item()

            metrics["f1"][pred_key] = f1_score(binary_preds_nozeros.cpu().numpy(), binary_truth_nozeros.cpu().numpy(), average='weighted')
            metrics["f1_has0"][pred_key] = f1_score(binary_preds.cpu().numpy(), binary_truth.cpu().numpy(), average='weighted')

            test_preds = total_preds.view(-1).cpu().detach().numpy()
            test_truth = targets_tens.view(-1).cpu().detach().numpy()

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            metrics["mae"][pred_key] = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
            metrics["corr"][pred_key] = np.corrcoef(test_preds, test_truth)[0][1]
            metrics["acc_7"][pred_key] = multiclass_acc(test_preds_a7, test_truth_a7)
            metrics["acc_5"][pred_key] = multiclass_acc(test_preds_a5, test_truth_a5)

        metrics = dict(metrics)

        return metrics

    def is_best(self, metrics = None, best_logs=None):
        if metrics is None:
            metrics = self.evaluate()

        validate_with = self.config.early_stopping.get("validate_with", "loss")
        if validate_with == "loss":
            is_best = (metrics["loss"]["total"] < best_logs["loss"]["total"])
        elif validate_with == "accuracy":
            is_best = (metrics["acc"]["combined"] > best_logs["acc"]["combined"])
        else:
            raise ValueError("self.agent.config.early_stopping.validate_with should be either loss or accuracy")
        return is_best

class General_Evaluator_Generative:
    """
    Evaluator for generative models.
    Stores features, losses, labels, samples and performs:
      - Feature similarity metrics
      - t-SNE visualization
      - WandB logging for t-SNE
    """

    def __init__(self, config: Any, total_instances: int, set: str = "val"):
        self.config = config
        self.set = set
        self.total_instances = total_instances
        self.reset()
        self.early_stop = False

    def reset(self) -> None:
        self.features: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.samples: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.losses: List[Dict[str, torch.Tensor]] = []
        self.processed_instances: int = 0

    def set_early_stop(self):
        self.early_stop = True

    def reset(self):
        self.losses = []
        self.preds = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.multi_supervised_w}
        self.features = defaultdict(list)
        self.labels = []
        self.processed_instances = 0

    def process(self, all_output: dict):

        logits = {pred: all_output["pred"][pred].cpu() for pred in all_output["pred"]}
        if self.set == "val":
            features = {feat: all_output["features"][feat].cpu() for feat in all_output["features"]}
        label = all_output["label"].cpu()
        loss = {l_i: all_output["loss"][l_i].detach().cpu() for l_i in all_output["loss"]}
        num_instances = label.shape[0]

        for pred_key in logits:
            if pred_key not in self.preds:
                continue
            assert (len(logits[pred_key].shape) == 2), "The shape of logits must be in format [bs, num_test_clips * num_test_crops, total_classes]"
            self.preds[pred_key].append(logits[pred_key])

        if self.set == "val":
            for feat_key in features:
                if feat_key not in self.features:
                    continue
                self.features[feat_key].append(features[feat_key])

        self.labels.append(label)

        self.processed_instances += num_instances
        self.losses.append(loss)

    def get_early_stop(self):
        return self.early_stop

    def enable_early_stop(self):
        self.early_stop = True

    def process(self, output: Dict[str, Any]) -> None:
        if "features" in output:
            for k, v in output["features"].items():
                self.features[k].append(v.detach().cpu())

        if "samples" in output:
            self.samples.append(output["samples"].detach().cpu())

        if "label" in output:
            self.labels.append(output["label"].detach().cpu())
            batch_size = output["label"].shape[0]
        else:
            if "features" in output:
                k0 = next(iter(output["features"]))
                batch_size = output["features"][k0].shape[0]
            else:
                batch_size = 1

        if "loss" in output:
            self.losses.append({k: v.detach().cpu() for k, v in output["loss"].items()})

        self.processed_instances += batch_size

    def mean_batch_loss(self) -> (Optional[Dict[str, float]], str):
        if not self.losses:
            return None, ""
        keys = self.losses[0].keys()
        mean_loss = {k: torch.stack([L[k] for L in self.losses]).mean().item() for k in keys}
        msg = " ".join(f"{k}: {v:.4f}" for k, v in mean_loss.items())
        return mean_loss, msg

    def evaluate(self, **eval_kwargs) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        mean_loss, _ = self.mean_batch_loss()
        if mean_loss is not None:
            metrics["loss"] = mean_loss

        features_val = {
            k: torch.cat(v, dim=0).numpy()
            for k, v in self.features.items()
        }

        if eval_kwargs.get("current_epoch", 0) > 5 and eval_kwargs["current_epoch"] % 10 == 0:
            metrics["feature_metrics"] = self._run_feature_metrics(features_val)
        if self.set == "test":
            metrics["feature_metrics"] = self._run_feature_metrics(features_val)
            labels_val = (
                torch.cat(self.labels, dim=0).numpy()
                if self.labels else None
            )
            self._run_tsne(features_val, labels_val)

        return metrics

    def _run_feature_metrics(self, features: Dict[str, np.ndarray]) -> Any:
        features_normalized = norm(features)
        results = compare_feature_spaces(features_normalized)
        print_feature_comparisons(results)
        return results

    def _run_tsne(self, features: Dict[str, np.ndarray], labels: Optional[np.ndarray]) -> None:
        try:
            import wandb
        except ImportError:
            wandb = None

        required = ("z1", "z2", "~z1", "~z2")
        if not all(k in features for k in required):
            return

        this_feature = {}
        for f in required:
            if len(features[f].shape)>2:
                this_feature[f] = features[f].mean(axis=2)
            else:
                this_feature[f] = features[f]

        fig = tsne_plot(
            this_feature["z1"],
            this_feature["z2"],
            this_feature["~z1"],
            this_feature["~z2"],
            labels,
            return_fig=True
        )

        if wandb is not None and wandb.run is not None:
            wandb.log({"tsne/validation_embeddings": wandb.Image(fig)})
        else:
            fig.show()
    def is_best(self, metrics = None, best_logs=None):
        if metrics is None:
            metrics = self.evaluate()

        is_best = (metrics["loss"]["total"] < best_logs["loss"]["total"])

        return is_best

def norm(features):
    new_features = {k: v.mean(axis=-1) if len(v.shape)>2 else v for k, v in features.items() }
    return new_features

def tsne_plot(z1, z2, z1_tilde, z2_tilde, y, perplexity=30, separate_tsne=True, return_fig=False):
    from openTSNE import TSNE
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    ts1 = TSNE(n_components=2, perplexity=perplexity).fit(z1)
    ts2 = TSNE(n_components=2, perplexity=perplexity).fit(z2)

    z1_e = np.asarray(ts1)
    z2_e = np.asarray(ts2)

    if separate_tsne:
        z1_t_e = TSNE(n_components=2, perplexity=perplexity).fit(z1_tilde)
        z2_t_e = TSNE(n_components=2, perplexity=perplexity).fit(z2_tilde)
    else:
        z1_t_e = ts1.transform(z1_tilde)
        z2_t_e = ts2.transform(z2_tilde)

    cmap = ListedColormap([
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"
    ])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    titles = ["z1", "z2", "z1_tilde", "z2_tilde"]
    data = [z1_e, z2_e, z1_t_e, z2_t_e]

    for ax, emb, title in zip(axs.ravel(), data, titles):
        ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap, s=8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
