"""CREMA-D backbones and the fusion models used by the paper.

Only the model classes referenced by the public configs are kept:
ResNet audio/visual encoders, FactorCL_Uni, and the Base / Base-Ensemble / MCR
fusion models. Earlier experimental variants (image-generation, captioning,
IHA, AGM/MLA/OGM, MCR noise/zero/3D variants, alternative backbones) were removed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import wandb

from synib.models.model_utils.backbone import resnet18
from synib.models.model_utils.fusion_gates import *  # noqa: F401,F403  (FiLM, GatedFusion, ...)
from synib.models.vlm.synib_mask_model import TF_Fusion_Transformer

try:
    from synib.mydatasets.Factor_CL_Datasets.MultiBench.unimodals.common_models import Transformer
except Exception:  # pragma: no cover
    Transformer = None

class Audio_ResNet(nn.Module):
    def __init__(self, args, encs):
        super(Audio_ResNet, self).__init__()

        self.args = args
        print(args)
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1

        # self.fusion_module = ConcatFusion(output_dim=n_classes)
        # self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        if args.get("cls_type", None) == "tf":
            self.aclassifier = TF_Fusion_Transformer(512, d_model, 2, num_classes)
        else:
            self.aclassifier = nn.Linear(512, num_classes)

    def forward_uni(self, x, na_x=None, **kwargs):
        detach_it = False
        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            detach_it = True

        if self.args.get("cls_type", None) == "tf":
            this_input = na_x
            if detach_it: this_input = this_input.detach()
            pred_a = self.aclassifier(this_input)
        else:
            this_input = x
            if detach_it: this_input = this_input.detach()
            pred_a = self.aclassifier(this_input)

        return pred_a

    def forward(self, x, **kwargs):

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # pred_a = self.common_fc(a)

        audio_feat = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(audio_feat, 1)
        a = torch.flatten(a, 1)
        na_a = audio_feat.flatten(start_dim=2).permute(0,2,1)
        if "detach_enc0" in kwargs and kwargs["detach_enc0"]:
            a = a.detach()
            na_a = na_a.detach()

        pred_a = self.forward_uni(a, na_a, **kwargs)

        return {"preds": {"combined": pred_a}, "features": {"combined": a}, "nonaggr_features":{"combined": na_a}}
class Video_ResNet(nn.Module):
    def __init__(self, args, encs):
        super(Video_ResNet, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        modality = args.get("modality", "visual")
        self.visual_net = resnet18(modality=modality)
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, weights='ResNet18_Weights.DEFAULT') # , weights='ResNet18_Weights.DEFAULT'

        # self.vclassifier = nn.Linear(512, num_classes)
        # self.vclassifier = nn.Linear(1000, num_classes)

        if args.get("cls_type", None) == "tf":
            self.vclassifier = TF_Fusion_Transformer(512, d_model, 2, num_classes)
        else:
            self.vclassifier = nn.Sequential(nn.Linear(d_model, num_classes))

    def forward_uni(self, x, na_x=None, **kwargs):
        detach_it = False
        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            detach_it = True

        if self.args.get("cls_type", None) == "tf":
            this_input = na_x
            if detach_it: this_input = this_input.detach()
            pred_v = self.vclassifier(this_input)
        else:
            this_input = x
            if detach_it: this_input = this_input.detach()
            pred_v = self.vclassifier(this_input)


        return pred_v

    def forward(self, x, **kwargs):


        v = self.visual_net(x[1])
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        video_feat = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(video_feat, 1)
        v = torch.flatten(v, 1)
        na_v = video_feat.flatten(start_dim=2).permute(0,2,1)

        if "detach_enc1" in kwargs and kwargs["detach_enc1"]:
            v = v.detach()
            na_v = na_v.detach()


        pred_v = self.forward_uni(v, na_v, **kwargs)

        return {"preds":{"combined":pred_v}, "features":{"combined":v}, "nonaggr_features":{"combined": na_v}}

class FactorCL_Uni(nn.Module):
    def __init__(self, args, encs):
        super(FactorCL_Uni, self).__init__()

        self.args = args
        n_features = args.get("n_features", 100)
        hidden_size = args.get("hidden_size", 100)

        self.enc = Transformer(n_features, hidden_size)
        self.pred_fc = nn.Linear(hidden_size, args.num_classes)

    def forward_uni(self, x, na_x=None, **kwargs):
        detach_it = kwargs.get("detach_pred", False)
        this_input = x.detach() if detach_it else x
        return self.pred_fc(this_input)

    def forward(self, x, **kwargs):

        x1 = x[self.args.modality]
        feat, feat_nonaggr = self.enc(x1)
        pred = self.pred_fc(feat)

        return {"preds": {"combined": pred}, "features": {"combined": feat}, "nonaggr_features": {"combined": feat_nonaggr}}



class MCR_Model(nn.Module):
    def __init__(self, args, encs):
        super(MCR_Model, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(d_model, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(d_model, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion_Transformer(input_dim=d_model, dim=d_model, layers=args.get("fusion_layers", 2), output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

        if self.args.bias_infusion.get("lib", 0) > 0:
            self.fc_yz = nn.Sequential(
                nn.Linear(num_classes, d_model, bias=False),
                nn.ReLU(),
                nn.Linear(d_model, d_model*2, bias=False),
            )

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, label):

        batch_size = label.size(0)
        shuffle_data = []
        random_shuffling = True
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:

                if self.args.bias_infusion.shuffle:
                    shuffle_idx = torch.randperm(batch_size)
                    if "rsl" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(label[shuffle_idx] == label)
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    elif "rsi" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(shuffle_idx == torch.arange(batch_size))
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    else:
                        nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                else:
                    nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                    shuffle_idx = torch.arange(batch_size)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})
        elif "samelabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li == lj and i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "difflabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li != lj:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "alllabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        return shuffle_data

    def shuffle_data(self, x, pred, label):
        if len(label.shape)>1:
            label = label.flatten()

        if not self.args.bias_infusion.get("training_mode", False):
            self.eval()

        a, v, pred_aa, pred_vv = self._get_features(x)

        shuffle_data = self.shuffle_ids(label)

        feat_dict = [ i for i in ["features", "nonaggr_features"] if i in a.keys() and i in v.keys() ]

        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in feat_dict}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in feat_dict}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in feat_dict}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in feat_dict}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label_shuffled = torch.concatenate([label[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        if not self.args.bias_infusion.get("training_mode", False):
            self.train()

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)


        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}}

        if self.training:
            if self.args.bias_infusion.get("lib", 0) > 0:
                pred_feat = self.fc_yz(pred.detach())
                combined_features = torch.cat([a["features"]["combined"], v["features"]["combined"]], dim=1)
                CMI_yz_Loss = torch.nn.MSELoss()(combined_features, pred_feat) * self.args.bias_infusion.get("lib", 0)
                output["losses"] = {"CMI_yz_Loss": CMI_yz_Loss}

            if self.args.bias_infusion.get("l", 0) != 0:

                sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled = self.shuffle_data( x, pred, kwargs["label"])

                pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
                pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, notwandb=True, **kwargs)
                output["preds"]["sa_detv"] = pred_dtv_sa
                output["preds"]["sa_deta"] = pred_dta_sa

                pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
                pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, notwandb=True, **kwargs)
                output["preds"]["sv_detv"] = pred_dtv_sv
                output["preds"]["sv_deta"] = pred_dta_sv

                pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), notwandb=True, **kwargs)
                pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), notwandb=True, **kwargs)

                output["preds"]["sv"] = pred_sv
                output["preds"]["sa"] = pred_sa

                output["preds"]["ncombined"] = n_pred

                output["preds"]["n_label"] = n_label
                output["preds"]["n_label_shuffled"] = n_label_shuffled

        return output
class Base_Ensemble_Model(nn.Module):
    def __init__(self, args, encs):
        super(Base_Ensemble_Model, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.num_classes = args.num_classes
        self.norm_decision = args.get("norm_decision", False)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.norm_decision == "batch_norm":
            self.norm_0 = nn.BatchNorm1d(self.num_classes , track_running_stats=False)
            self.norm_1 = nn.BatchNorm1d(self.num_classes , track_running_stats=False)
        elif self.norm_decision == "instance_norm":
            self.norm_0 = nn.InstanceNorm1d(self.num_classes , track_running_stats=False)
            self.norm_1 = nn.InstanceNorm1d(self.num_classes , track_running_stats=False)
        elif self.norm_decision == "softmax":
            self.norm_0 = nn.Softmax(dim=1)
            self.norm_1 = nn.Softmax(dim=1)


    def _get_features(self, x):
        if self.enc_0.args.get("freeze_encoder", False):
            self.enc_0.eval()
        if self.enc_1.args.get("freeze_encoder", False):
            self.enc_1.eval()

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["preds"]["combined"], v["preds"]["combined"], a["features"]["combined"], v["features"]["combined"]

    def forward(self, x, **kwargs):

        pred_a, pred_v, a, v = self._get_features(x)

        if self.norm_decision == "standardization":
            pred_a = (pred_a - pred_a.mean())/pred_a.std()
            pred_v = (pred_v - pred_v.mean())/pred_v.std()
            pred = pred_a + pred_v

        elif self.norm_decision == "batch_norm" or self.norm_decision == "instance_norm":

            pred_a = self.norm_0(pred_a)
            pred_v = self.norm_1(pred_v)
            pred = pred_a + pred_v

        elif self.norm_decision == "softmax":

            pred_a = self.norm_0(pred_a)
            pred_v = self.norm_1(pred_v)
            pred = torch.nn.functional.softmax(pred_a, dim=1) + torch.nn.functional.softmax(pred_v, dim=1)
        else:
            pred = pred_a + pred_v

        # if a.shape != v.shape:
        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features":{}}
        # return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}
class Base_Model(nn.Module):
    def __init__(self, args, encs):
        super(Base_Model, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.cls_type = args.get("cls_type", "linear")

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear" or self.cls_type == "linear_stopgrad" or self.cls_type =="linear_ogm" or self.cls_type == "linear_ogm_multi":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
        elif self.cls_type == "dec":
            pass

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion_Transformer(input_dim=d_model, dim=d_model, layers=args.get("fusion_layers", 2), output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")
    def _get_features(self, x, detach_pred=False):

        a = self.enc_0(x, detach_pred=detach_pred)
        v = self.enc_1(x, detach_pred=detach_pred)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def forward(self, x, **kwargs):

        if self.cls_type == "linear_stopgrad":
            a, v, pred_aa, pred_vv = self._get_features(x, detach_pred=True)
        else:
            a, v, pred_aa, pred_vv = self._get_features(x)

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a["features"]["combined"], dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v["features"]["combined"], dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v
        elif self.cls_type == "linear_ogm":
            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2
            pred = pred_a + pred_v
            pred_aa = pred_a.detach()
            pred_vv = pred_v.detach()
        elif self.cls_type == "linear_ogm_multi":
            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2
            pred = pred_a + pred_v
            pred_aa = pred_a
            pred_vv = pred_v
        elif self.cls_type in ("linear", "highlynonlinear", "nonlinear", "linear_stopgrad"):

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

            pred = pred_a + pred_v
        else:
            pred = pred_aa + pred_vv

        if self.cls_type == "film" or self.cls_type == "gated":
            pred = self.common_fc([a["features"]["combined"], v["features"]["combined"]])
        elif self.cls_type == "tf":
            pred = self.common_fc([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]])
        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)
        if (self.args.bias_infusion.method == "OGM" or self.args.bias_infusion.method == "OGM_GE" or self.args.bias_infusion.method == "MSLR") and self.cls_type!="dec":
            pred_aa = pred_a
            pred_vv = pred_v

        return {"preds":{"combined":pred,
                         "c":pred_aa,
                         "g":pred_vv
                         },
                "features": {"c": a["features"]["combined"],
                             "g": v["features"]["combined"]}}
