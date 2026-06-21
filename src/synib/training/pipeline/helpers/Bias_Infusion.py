import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import copy
import math
import wandb
from pytorch_metric_learning.losses import NTXentLoss
from synib.utils.optimization.min_norm_solver import MinNormSolver
from synib.utils.optimization.gs_plugin import GSPlugin
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from synib.utils.data.to_device import to_device

def pick_bias_infuser(agent):
    method = agent.config.model.args.get("bias_infusion", {}).get("method", False)
    if method == "MMPareto":
        bi = Bias_Infusion_MMPareto(agent)
    elif method == "DnR":
        bi = Bias_Infusion_DnR(agent)
    elif method == "ReconBoost":
        bi = Bias_Infusion_ReconBoost(agent)
    elif method == "MCR":
        bi = Bias_Infusion_MCR(agent)
    else:
        bi = General_Bias_Infusion(agent)
    return bi

class General_Bias_Infusion():
    def __init__(self, agent):
        self.agent = agent

        super(General_Bias_Infusion, self).__init__()

    def before_backward(self, total, output_losses, **kwargs):
        return total, output_losses, False

    def on_backward_end(self, **kwargs):
        return

    def on_epoch_begin(self, **kwargs):
        pass

    def plot_bias(self, **kwargs):
        pass


class Bias_Infusion_MMPareto(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_MMPareto, self).__init__(agent)
        logging.info("Bias Infusion MMPareto is being employed")
        self._initialize_logs_n_utils()
        self.cosine_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def _initialize_logs_n_utils(self):
        pass
    
    def before_backward(self, total, output_losses, **kwargs):
        if not self.agent.config.model.args.bias_infusion.use: return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs[
            "current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:
            print(output_losses)
            loss_mm = output_losses["ce_loss_combined"]
            loss_a = output_losses["ce_loss_c"]
            loss_v = output_losses["ce_loss_g"]

            losses = [loss_mm, loss_a, loss_v]
            all_loss = ['both', 'audio', 'visual']

            grads_visual = defaultdict(dict)
            grads_audio = defaultdict(dict)

            for idx, loss_type in enumerate(all_loss):
                loss = losses[idx]
                loss.backward(retain_graph=True)
                if (loss_type == 'visual'):
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None: continue
                        if ("mod1" in name or "fc_1" in name or "enc_1" in name) and name in grads_visual["both"]:
                            grads_visual[loss_type][name] = parms.grad.data.clone()
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [grads_visual[loss_type][name].flatten()
                         for name, parms in self.agent.model.named_parameters()
                         if ("mod1" in name or "enc_1" in name)
                         and parms.grad is not None
                         and name in grads_visual["both"]])
                elif (loss_type == 'audio'):
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None: continue
                        if ("mod0" in name or "enc_0" in name) and name in grads_audio["both"]:
                            grads_audio[loss_type][name] = parms.grad.data.clone()
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [grads_audio[loss_type][name].flatten()
                         for name, parms in self.agent.model.named_parameters()
                         if ("mod0" in name or "enc_0" in name)
                         and parms.grad is not None
                         and name in grads_audio["both"]])
                else:
                    for name, parms in self.agent.model.named_parameters():
                        if parms.grad is None: continue
                        if "mod0" in name or "enc_0" in name:
                            grads_audio[loss_type][name] = parms.grad.data.clone()
                        if "mod1" in name or "enc_1" in name:
                            grads_visual[loss_type][name] = parms.grad.data.clone()
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [grads_visual[loss_type][name].flatten() for name, parms in
                         self.agent.model.named_parameters() if
                         ("mod1" in name or "enc_1" in name) and parms.grad is not None])
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [grads_audio[loss_type][name].flatten() for name, parms in
                         self.agent.model.named_parameters() if
                         ("mod0" in name or "enc_0" in name) and parms.grad is not None])
                self.agent.optimizer.zero_grad()

            audio_k, visual_k = self._compute_ratio(grads_audio, grads_visual)
            total = loss_mm + loss_a + loss_v
            total.backward()
            gamma = self.agent.config.model.args.bias_infusion.alpha
            self._equalize_gradients(grads_audio, grads_visual, audio_k, visual_k, gamma)

            self.agent.optimizer.step()

            wandb_output = {"ratio": {"audio_k": audio_k, "visual_k": visual_k}}
            wandb.log(wandb_output)

            return total, output_losses, True

    def _compute_ratio(self, grads_audio, grads_visual):
        this_cos_audio = self.cosine_sim(grads_audio['both']["concat"], grads_audio['audio']["concat"])
        this_cos_visual = self.cosine_sim(grads_visual['both']["concat"], grads_visual['visual']["concat"])

        audio_task = ['both', 'audio']
        visual_task = ['both', 'visual']
        
        audio_k = [0, 0]
        visual_k = [0, 0]

        if (this_cos_audio > 0):
            audio_k[0] = 0.5
            audio_k[1] = 0.5
        else:
            audio_k, min_norm = MinNormSolver.find_min_norm_element(
                [list(grads_audio[t].values()) for t in audio_task])
        if (this_cos_visual > 0):
            visual_k[0] = 0.5
            visual_k[1] = 0.5
        else:
            visual_k, min_norm = MinNormSolver.find_min_norm_element(
                [list(grads_visual[t].values()) for t in visual_task])
        return audio_k, visual_k

    def _equalize_gradients(self, grads_audio, grads_visual, audio_k, visual_k, gamma):
        for name, param in self.agent.model.named_parameters():
            if param.grad is not None:
                if ("mod0" in name or "fc_0" in name or "enc_0" in name) and name in grads_audio['both']:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * audio_k[0] * grads_audio['both'][name] + 2 * audio_k[1] * \
                               grads_audio['audio'][
                                   name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if (diff > 1):
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma

                if ("mod1" in name or "fc_1" in name or "enc_1" in name) and name in grads_visual['both']:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * visual_k[0] * grads_visual['both'][name] + 2 * visual_k[1] * \
                               grads_visual['visual'][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if (diff > 1):
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma


class Bias_Infusion_ReconBoost(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_ReconBoost, self).__init__(agent)
        logging.info("Bias Infusion ReconBoost is being employed")
        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        pass

    def get_stage(self, epoch, epoch_stages, ensemble_stages):
        cycle_length = 2 * epoch_stages + 2 * ensemble_stages  # Total length of one full cycle
        position = epoch % cycle_length  # Position within the cycle

        if position < epoch_stages:
            return 0
        elif position < epoch_stages + ensemble_stages:
            return 1
        elif position < 2 * epoch_stages + ensemble_stages:
            return 2
        else:
            return 1  # Last phase of the cycle

    def before_backward(self, total, output_losses, **kwargs):
        if not self.agent.config.model.args.bias_infusion.use: return

        if self.agent.config.model.args.bias_infusion.starting_epoch <= self.agent.logs[
            "current_epoch"] <= self.agent.config.model.args.bias_infusion.ending_epoch:

            target = torch.zeros(kwargs["output"]["preds"]["c"].shape[0], self.agent.config.model.args.num_classes).cuda().scatter_(1, kwargs["label"].view(-1, 1), 1)

            stages = self.get_stage(self.agent.logs["current_epoch"], self.agent.config.model.args.bias_infusion.epoch_stages, self.agent.config.model.args.bias_infusion.ensemble_stages)

            if stages == 0 or stages == 2:
                modality = stages // 2

                if modality == 0:
                    out_obj = kwargs["output"]["preds"]["c"]
                elif modality == 1:
                    out_obj = kwargs["output"]["preds"]["g"]
                out_join = kwargs["output"]["preds"]["combined"]

                boosting_loss = - self.agent.config.model.args.bias_infusion.weight1 * (target * out_obj.log_softmax(1)).mean(-1) \
                                + self.agent.config.model.args.bias_infusion.weight2 * (target * out_join.detach().softmax(1) * out_obj.log_softmax(1)).mean(-1)

                for name, param in self.agent.model.named_parameters():
                    if modality == 0 and ("mod1" in name or "fc_1" in name or "enc_1" in name):
                        param.requires_grad = False
                    elif modality == 1 and ("mod0" in name or "fc_0" in name or "enc_0" in name):
                        param.requires_grad = False

                self.agent.model.zero_grad()

                if self.agent.config.model.args.bias_infusion.use_ga:
                    if self.agent.logs["current_epoch"]//self.agent.config.model.args.bias_infusion.epoch_stages == 0:
                        loss = boosting_loss
                    else:
                        if modality == 0:
                            pre_out_obj = kwargs["output"]["preds"]["g"]
                        elif modality == 1:
                            pre_out_obj = kwargs["output"]["preds"]["c"]
                        ga_loss = nn.MSELoss()(out_obj.detach().softmax(1), pre_out_obj.detach().softmax(1))  ## ga loss
                        loss = boosting_loss + self.agent.config.model.args.bias_infusion.alpha * ga_loss
                    loss.mean().backward()
                else:
                    boosting_loss.mean().backward()

                self.agent.optimizer.step()
                for name, param in self.agent.model.named_parameters():
                    if modality == 0 and ("mod1" in name or "fc_1" in name or "enc_1" in name):
                        param.requires_grad = True
                    elif modality == 1 and ("mod0" in name or "fc_0" in name or "enc_0" in name):
                        param.requires_grad = True

                output_losses = {"recon": boosting_loss.mean()}

            else:

                output_losses["ce_loss_combined"].backward()
                self.agent.optimizer.step()

            return total, output_losses, True

    def _compute_ratio(self, grads_audio, grads_visual):
        this_cos_audio = F.cosine_similarity(grads_audio['both']["concat"], grads_audio['audio']["concat"], dim=0)
        this_cos_visual = F.cosine_similarity(grads_visual['both']["concat"], grads_visual['visual']["concat"], dim=0)

        audio_task = ['both', 'audio']
        visual_task = ['both', 'visual']

        audio_k = [0, 0]
        visual_k = [0, 0]

        if (this_cos_audio > 0):
            audio_k[0] = 0.5
            audio_k[1] = 0.5
        else:
            audio_k, min_norm = MinNormSolver.find_min_norm_element(
                [list(grads_audio[t].values()) for t in audio_task])
        if (this_cos_visual > 0):
            visual_k[0] = 0.5
            visual_k[1] = 0.5
        else:
            visual_k, min_norm = MinNormSolver.find_min_norm_element(
                [list(grads_visual[t].values()) for t in visual_task])
        return audio_k, visual_k

    def _equalize_gradients(self, grads_audio, grads_visual, audio_k, visual_k, gamma):
        for name, param in self.agent.model.named_parameters():
            if param.grad is not None:
                if ("mod0" in name or "fc_0" in name or "enc_0" in name) and name in grads_audio['both']:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * audio_k[0] * grads_audio['both'][name] + 2 * audio_k[1] * \
                               grads_audio['audio'][
                                   name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if (diff > 1):
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma

                if ("mod1" in name or "fc_1" in name or "enc_1" in name) and name in grads_visual['both']:
                    three_norm = torch.norm(param.grad.data.clone())
                    new_grad = 2 * visual_k[0] * grads_visual['both'][name] + 2 * visual_k[1] * \
                               grads_visual['visual'][name]
                    new_norm = torch.norm(new_grad)
                    diff = three_norm / new_norm
                    if (diff > 1):
                        param.grad = diff * new_grad * gamma
                    else:
                        param.grad = new_grad * gamma

class Bias_Infusion_DnR(General_Bias_Infusion):
    def __init__(self, agent):
        super(Bias_Infusion_DnR, self).__init__(agent)
        logging.info("Bias Infusion DnR is being employed")
        self._initialize_logs_n_utils()
        self.alpha = self.agent.config.model.args.bias_infusion.alpha
        self.reinit_epoch = self.agent.config.model.args.bias_infusion.reinit_epoch
        self.kmepoch = self.agent.config.model.args.bias_infusion.kmepoch

    def _initialize_logs_n_utils(self):
        self.checkpoint_model = None
        self.flag_reinit = 0

    def get_feature(self, args, epoch, this_dataloader, desc="Train"):
        self.agent.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(this_dataloader),
                        total=len(this_dataloader),
                        desc=desc,
                        leave=False,
                        disable=True,
                        position=1)
            feature_dict = defaultdict(list)
            labels = []
            for batch_idx, served_dict in pbar:

                if type(served_dict) == tuple:
                    served_dict = {"data":{"c":served_dict[0][0], "f":served_dict[0][1], "g":served_dict[0][2]}, "label":served_dict[3].squeeze(dim=1)}
                    if self.agent.config.get("task", "classification") == "classification" and len(served_dict["label"][served_dict["label"]==-1])>0:
                        served_dict["label"][served_dict["label"] == -1] = 0

                data = to_device(served_dict["data"], "cuda")

                label = served_dict["label"].squeeze().type(torch.LongTensor).cuda()

                output = self.agent.model(data, label=label)

                for view in output["features"]:
                    feature_dict[view].append(output["features"][view].detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())
                del output, label, data
                pbar_message = "Validation batch {0:d}/{1:d}".format(batch_idx, len(this_dataloader) - 1 )
                pbar.set_description(pbar_message)
                pbar.refresh()

        labels = np.concatenate(labels)
        for view in feature_dict:
            feature_dict[view] = np.concatenate(feature_dict[view])
        return feature_dict, labels

    def purity_score(self, y_true, y_pred):

        y_voted_labels = np.zeros(y_true.shape)
        labels = np.unique(y_true)
        ordered_labels = np.arange(labels.shape[0])
        for k in range(labels.shape[0]):
            y_true[y_true == labels[k]] = ordered_labels[k]

        labels = np.unique(y_true)

        bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

        for cluster in np.unique(y_pred):
            hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
            winner = np.argmax(hist)
            y_voted_labels[y_pred == cluster] = winner

        return accuracy_score(y_true, y_voted_labels)

    def reinit_score(self, args, train_features, train_label, val_features, val_label):

        all_feature = [train_features["c"], val_features["g"], train_features["g"], val_features["g"]]
        stages = ['train_audio', 'val_audio', 'train_visual', 'val_visual']
        all_purity = []
        print('%%%%%%%%%%%%%%%%%%%%%%%%')
        for idx, fea in enumerate(all_feature):
            result = fea
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            result = scaler.fit_transform(result)
            y_pred = KMeans(n_clusters=args.n_classes, random_state=0, n_init=10).fit_predict(result)

            if (stages[idx][:5] == 'train'):
                purity = self.purity_score(np.array(train_label), y_pred)
            else:
                purity = self.purity_score(np.array(val_label), y_pred)
            all_purity.append(purity)

            print('%s purity= %.4f' % (stages[idx], purity))

        purity_gap_audio = np.abs(all_purity[0] - all_purity[1])
        purity_gap_visual = np.abs(all_purity[2] - all_purity[3])

        weight_audio = torch.tanh(torch.tensor(self.alpha * purity_gap_audio))
        weight_visual = torch.tanh(torch.tensor(self.alpha * purity_gap_visual))

        print('weights audio: {:.4f}, visual: {:.4f}'.format(weight_audio, weight_visual))

        return weight_audio, weight_visual, all_purity
    def reinit(self, args, weight_audio, weight_visual):

        print("Start reinit ... ")

        record_names_audio = []
        record_names_visual = []
        for name, param in self.agent.model.named_parameters():
            if ("mod0" in name or "fc_0" in name or "enc_0" in name):
                if ('conv' in name):
                    record_names_audio.append((name, param))
            elif ("mod1" in name or "fc_1" in name or "enc_1" in name):
                if ('conv' in name):
                    record_names_visual.append((name, param))

        for name, param in self.agent.model.named_parameters():
            if ("mod0" in name or "fc_0" in name or "enc_0" in name):
                init_weight = self.checkpoint_model[name]
                current_weight = param.data
                new_weight = weight_audio * init_weight + (1 - weight_audio).cuda() * current_weight
                param.data = new_weight
            elif ("mod1" in name or "fc_1" in name or "enc_1" in name):
                init_weight = self.checkpoint_model[name]
                current_weight = param.data
                new_weight = weight_visual * init_weight + (1 - weight_visual).cuda() * current_weight
                param.data = new_weight

        return self.agent.model

    def on_epoch_begin(self, **kwargs):
        if not self.agent.config.model.args.bias_infusion.use: return

        epoch = self.agent.logs["current_epoch"]

        if self.checkpoint_model is None:
            self.checkpoint_model = copy.deepcopy(self.agent.model.state_dict())
        if self.agent.config.model.args.bias_infusion.starting_epoch <= epoch <= self.agent.config.model.args.bias_infusion.ending_epoch:
            args = self.agent.config.model.args.bias_infusion

            if(( epoch % self.reinit_epoch == 0)&(epoch>0)):
                self.flag_reinit+=1
                if(self.flag_reinit<=self.kmepoch):
                    train_features, train_label = self.get_feature(args, epoch, self.agent.data_loader.train_loader)
                    val_features, val_label = self.get_feature(args, epoch, self.agent.data_loader.valid_loader)
                    weight_audio, weight_visual, all_purity = self.reinit_score(args, train_features,train_label,val_features,val_label)
                    self.agent.model = self.reinit(args, weight_audio, weight_visual)
                    wandb_output = {"ratio": {"audio_k": weight_audio, "visual_k": weight_visual, "audio_purity_train": all_purity[0],"audio_purity_train": all_purity[1], "visual_purity_train": all_purity[2], "visual_purity_val": all_purity[3]}}
                    wandb.log(wandb_output)


class Bias_Infusion_MCR(General_Bias_Infusion):
    """
    NGMine = Norm-Gradient my version
    """

    def __init__(self, agent):
        super(Bias_Infusion_MCR, self).__init__(agent)

        self._initialize_logs_n_utils()

    def _initialize_logs_n_utils(self):
        self.losses = []
        self.agent.logs["reg_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.agent.logs["ratio_logs"] = defaultdict(list)
        self.bias_infuser = self.agent.config.model.args.get("bias_infusion", {})
        self.regby = self.bias_infuser.get("regby", "greedy")
        self.l = self.bias_infuser.get("l", 0)
        self.contr_coeff = self.bias_infuser.get("contr_coeff", False)
        self.temperature = self.bias_infuser.get("temperature", 0.5)

    def get_perturbed_gradients(self, output):

        def log_likelihood_ratio(logits_Z, logits_Z_hat):
            """
            Compute the negative average log likelihood ratio between two sets of logits.

            Args:
            - logits_Z (torch.Tensor): Logits from predictions using (X, Z), shape (batch_size, num_classes)
            - logits_Z_hat (torch.Tensor): Logits from predictions using (X, Z_hat), shape (batch_size, num_classes)
            - targets (torch.Tensor): True class labels, shape (batch_size,)

            Returns:
            - torch.Tensor: Difference of negative average log likelihood ratios
            """

            probs_Z = F.softmax(logits_Z, dim=1)
            probs_Z_hat = F.softmax(logits_Z_hat, dim=1)

            log_probs_Z = F.log_softmax(logits_Z, dim=1)
            log_probs_Z_hat = F.log_softmax(logits_Z_hat, dim=1)

            per_sample_avg_log_prob_Z_hat = torch.concatenate([(log_probs_Z_hat * probs_Z_hat).mean().unsqueeze(0)])
            per_sample_avg_log_prob_Z_hat_onlylog = torch.concatenate([(log_probs_Z_hat).mean().unsqueeze(0)])

            avg_log_prob_Z = -torch.mean(log_probs_Z * probs_Z)
            avg_log_prob_Z_hat = -torch.mean(per_sample_avg_log_prob_Z_hat)

            avg_log_prob_Z_hat_onlylog = -torch.mean(per_sample_avg_log_prob_Z_hat_onlylog)
            avg_log_prob_Z_onlylog = -torch.mean(log_probs_Z)

            difference = - avg_log_prob_Z + avg_log_prob_Z_hat
            difference_onlylog = - avg_log_prob_Z_onlylog + avg_log_prob_Z_hat_onlylog

            return difference_onlylog, avg_log_prob_Z_hat_onlylog, avg_log_prob_Z_onlylog

        def js_divergence(net_1_logits, net_2_logits):

            clip_value = 1e+7

            net_1_probs = F.softmax(torch.clamp(net_1_logits, -clip_value, clip_value), dim=1)
            net_2_probs = F.softmax(torch.clamp(net_2_logits, -clip_value, clip_value), dim=1)

            total_m = 0.5 * (net_1_probs + net_2_probs)

            clip_value = 1e-20
            total_m = torch.clamp(total_m, clip_value, 1).log()

            net_1_probs = torch.clamp(net_1_probs, clip_value, 1)
            net_2_probs = torch.clamp(net_2_probs, clip_value, 1)

            loss = 0.0

            loss += F.kl_div(total_m, net_1_probs, reduction="batchmean")
            loss += F.kl_div(total_m, net_2_probs, reduction="batchmean")
            if torch.isnan(loss):
                raise Exception("NaN detected in loss computation")
            return (0.5 * loss)

        def get_losses(predicted_logits_00, soft_labels_norm, cross_permod, label, label_shuffled, title):

            jsd_label = js_divergence(predicted_logits_00, label)
            jsd_pred = js_divergence(predicted_logits_00, soft_labels_norm)
            loglike_pred, entropy_y, norm_ent = log_likelihood_ratio(soft_labels_norm, predicted_logits_00)

            cross_permod["{}_yent".format(title)].append(entropy_y)
            cross_permod["{}_nll".format(title)].append(loglike_pred)
            cross_permod["{}_label".format(title)].append(jsd_label)
            cross_permod["{}_pred".format(title)].append(jsd_pred)

            return cross_permod

        cross_permod = defaultdict(list)
        modimport_permod = {"enc_0": [], "enc_1": []}

        if "sa_detv" in output["preds"] and "sa_deta" in output["preds"] and "sa" in output["preds"]:
            label = F.one_hot(output["preds"]["n_label"], num_classes=self.agent.config.model.args.num_classes)
            label_shuffled = F.one_hot(output["preds"]["n_label_shuffled"],
                                       num_classes=self.agent.config.model.args.num_classes)

            cross_permod = get_losses(output["preds"]["sa_detv"], output["preds"]["ncombined"], cross_permod,
                                      label=label, label_shuffled=label_shuffled, title="enc0_p0")
            cross_permod = get_losses(output["preds"]["sa_deta"], output["preds"]["ncombined"], cross_permod,
                                      label=label, label_shuffled=label_shuffled, title="enc1_p0")
            cross_permod = get_losses(output["preds"]["sa"], output["preds"]["ncombined"], cross_permod, label=label,
                                      label_shuffled=label_shuffled, title="all_p0")
            norm_acc = (torch.argmax(output["preds"]["ncombined"], dim=1) == output["preds"]["n_label"]).float().mean()
            acc_pert0 = (torch.argmax(output["preds"]["sa_detv"], dim=1) == output["preds"]["n_label"]).float().mean()
        else:
            cross_permod["enc0_p0_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc1_p0_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["all_p0_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc0_p0_nll"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc1_p0_nll"].append(torch.tensor(0.0).to(self.agent.device))

            norm_acc = torch.tensor(0.0).to(self.agent.device)
            acc_pert0 = torch.tensor(0.0).to(self.agent.device)

        if "sv_detv" in output["preds"] and "sv_deta" in output["preds"] and "sv" in output["preds"]:
            label = F.one_hot(output["preds"]["n_label"], num_classes=self.agent.config.model.args.num_classes)
            label_shuffled = F.one_hot(output["preds"]["n_label_shuffled"],
                                       num_classes=self.agent.config.model.args.num_classes)

            cross_permod = get_losses(output["preds"]["sv_deta"], output["preds"]["ncombined"], cross_permod,
                                      label=label, label_shuffled=label_shuffled, title="enc1_p1")
            cross_permod = get_losses(output["preds"]["sv_detv"], output["preds"]["ncombined"], cross_permod,
                                      label=label, label_shuffled=label_shuffled, title="enc0_p1")
            cross_permod = get_losses(output["preds"]["sv"], output["preds"]["ncombined"], cross_permod, label=label,
                                      label_shuffled=label_shuffled, title="all_p1")
            acc_pert1 = (torch.argmax(output["preds"]["sv_deta"], dim=1) == output["preds"]["n_label"]).float().mean()
        else:
            cross_permod["enc0_p1_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc1_p1_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["all_p1_pred"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc0_p1_nll"].append(torch.tensor(0.0).to(self.agent.device))
            cross_permod["enc1_p1_nll"].append(torch.tensor(0.0).to(self.agent.device))
            acc_pert1 = torch.tensor(0.0).to(self.agent.device)

        modimport_permod["enc_0"].append(acc_pert1 / norm_acc)
        modimport_permod["enc_1"].append(acc_pert0 / norm_acc)

        out = {name: torch.stack(cross_permod[name]).mean() for name in cross_permod}

        modimport_permod = {name: torch.stack(modimport_permod[name]).mean() for name in modimport_permod}

        return out, modimport_permod

    def before_backward(self, total, output_losses, w_loss, loss_fun, data, label, output, **kwargs):

        if not (self.bias_infuser.get("starting_epoch", 0) <= self.agent.logs["current_epoch"] <= self.bias_infuser.get(
                "ending_epoch", 1000)):
            return total, output_losses, False

        num_samples = self.agent.config.model.args.bias_infusion.num_samples

        pert_loss, import_permod = self.get_perturbed_gradients(output)

        l = self.l

        if self.regby == "greedy":
            reg_term = self.l * (- pert_loss["enc0_p0_pred"]
                            + pert_loss["enc1_p0_pred"]
                            - pert_loss["enc1_p1_pred"]
                            + pert_loss["enc0_p1_pred"])
            reg_term_0 = pert_loss["enc0_p0_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"]
        elif self.regby == "ind":
            reg_term = self.l * (- pert_loss["enc0_p0_pred"]
                            - pert_loss["enc1_p1_pred"])
            reg_term_0 = pert_loss["enc0_p0_pred"]
            reg_term_1 = pert_loss["enc1_p1_pred"]
        elif self.regby == "colab":
            reg_term_0 = pert_loss["all_p0_pred"]
            reg_term_1 = pert_loss["all_p1_pred"]
            reg_term = self.l * (- reg_term_0 - reg_term_1)

        if self.contr_coeff:
            if self.bias_infuser.get("contr_type", "label") == "label":
                if len(label.shape) > 1:
                    label_partitions = 30
                    label = label.flatten()
                    cca_loss = []
                    for each_part in range(label_partitions):
                        from_id = int(len(label) / label_partitions) * each_part
                        to_id = int(len(label) / label_partitions) * (each_part + 1)
                        if each_part == label_partitions - 1:
                            to_id = len(label)
                        cca_loss.append(
                            nt_xent_loss(output["features"]["c"][from_id:to_id], output["features"]["g"][from_id:to_id],
                                         label=label[from_id:to_id],
                                         temperature=self.temperature).unsqueeze(0))
                    cca_loss = torch.cat(cca_loss).mean()
                else:
                    if "f" in output["features"]:
                        cca_loss = nt_xent_loss(output["features"]["c"], output["features"]["g"],
                                                output["features"]["f"], label=label,
                                                temperature=self.temperature)
                    else:
                        cca_loss = nt_xent_loss(output["features"]["c"], output["features"]["g"], label=label,
                                                temperature=self.temperature)

            elif self.bias_infuser.get("contr_type", "label") == "pure":
                if "f" in output["features"]:
                    cca_loss = nt_xent_loss(output["features"]["c"], output["features"]["g"], output["features"]["f"],
                                            temperature=self.temperature)
                else:
                    cca_loss = nt_xent_loss(output["features"]["c"], output["features"]["g"],
                                            temperature=self.temperature)

            cca_loss_reg = self.contr_coeff * cca_loss

        reg_terms = {}

        if "reg_term_0" in locals() and "reg_term_1" in locals():
            reg_terms["reg_term_0"] = reg_term_0
            reg_terms["reg_term_1"] = reg_term_1
            output_losses["reg_term_0"] = reg_term_0
            output_losses["reg_term_1"] = reg_term_1

        wandb.log({
            "perturb": pert_loss,
            "reg_terms": reg_terms
        }, step=self.agent.logs["current_step"] + 1)

        if "reg_term" in locals():
            output_losses["reg_term"] = reg_term
            total = total + reg_term
        if self.contr_coeff:
            output_losses["cca"] = cca_loss
            total = total + cca_loss_reg

        return total, output_losses, False
