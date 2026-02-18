
from synib.models.model_utils.fusion_gates import *
from synib.models.VAVL_git.VAVL.conformer.model import Conformer
from synib.models.crema_d.crema_backbone_fusion_models import TF_Fusion
from pytorch_metric_learning.losses import NTXentLoss
from torch.nn.utils import spectral_norm as SN
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from transformers import AutoTokenizer
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


class TF_Proc(nn.Module):
    def __init__(self, input_dim, dim, layers, output_dim):
        super(TF_Proc, self).__init__()
        self.common_net = Conformer(
                            input_dim=input_dim,
                            encoder_dim=dim,
                            num_encoder_layers=layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)

        self.common_fc = nn.Linear(dim, output_dim)


    def forward(self, x, **kwargs):
        x_0 = x.permute(0,2,1)

        x_0 = self.cls_token.repeat(x_0.shape[0], x_0.shape[1], 1) + x_0

        if "detach_feat" in kwargs and kwargs["detach_feat"]:
            x_0 = x_0.detach()

        feat_mm = torch.concatenate([self.cls_token.repeat(x_0.shape[0], 1, 1), x_0], dim=1)
        feat_mm = self.common_net(feat_mm)
        aggr_feat_mm = feat_mm[:,0]

        pred = self.common_fc(aggr_feat_mm)
        if kwargs.get("return_all", False):
            return pred, aggr_feat_mm, feat_mm
        else:
            return pred

def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_clamp_loss(z12, z12_hat):
    cos_sim = F.cosine_similarity(z12, z12_hat, dim=-1)
    # want similarity low → hinge on (1 - cos_sim)
    return torch.clamp(cos_sim, min=0)

def nt_xent_loss(z_x, z_y, z_k=None, label=None, temperature=0.5):
    # z_x = F.normalize(z_x, p=2, dim=1)
    # z_y = F.normalize(z_y, p=2, dim=1)
    z = torch.cat([z_x, z_y], dim=0)
    if label is not None:
        labels = torch.cat([label, label], dim=0)
    else:
        labels = torch.cat([torch.arange(z_x.shape[0]), torch.arange(z_y.shape[0])], dim=0)
    if z_k is not None:
        z_k = F.normalize(z_k, p=2, dim=1)
        z = torch.cat([z, z_k], dim=0)
        if label is not None:
            labels = torch.cat([label, label, label], dim=0)
        else:
            labels = torch.cat([torch.arange(z_x.shape[0]), torch.arange(z_y.shape[0]), torch.arange(z_k.shape[0])],
                               dim=0)

    loss = NTXentLoss(temperature=temperature)(z, labels)

    return loss


class Fusion_Synprom(nn.Module):
    def __init__(self, args, encs):
        super(Fusion_Synprom, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)

        self.count_trainingsteps = 0

        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(d_model, num_classes)
            )
            # self.fc_yz = nn.Linear(num_classes, d_model*2, bias=False)
            self.fc_yz1z2 = nn.Sequential(
                nn.Linear(num_classes + d_model * 2, d_model * 2, bias=False),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model, bias=False),
            )

            self.fc_z1z2 = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2, bias=False),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model, bias=False),
            )
        elif self.cls_type == "conformer":

            self.common_fc_1 =TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6, output_dim=d_model)
            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(d_model, num_classes)
            )
            self.lin_pred = nn.Linear(num_classes, d_model)

            self.fc_yz1z2 = TF_Proc(
                input_dim=d_model,
                dim=d_model,
                layers=2, output_dim=d_model)

            self.fc_z1z2 = TF_Proc(
                input_dim=d_model,
                dim=d_model,
                layers=2, output_dim=d_model)
        else:
            raise ValueError("Unknown cls_type")


        # self.fc_1_mu = nn.Linear(d_model,  self.latent_dim, bias=False)
        # self.fc_0_logvar = nn.Linear(d_model, self.latent_dim, bias=False)
        # self.fc_1_logvar = nn.Linear(d_model, self.latent_dim, bias=False)


    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)

        joint_a = self.enc_2(x, **kwargs)
        joint_v = self.enc_3(x, **kwargs)

        return a, v, joint_a, joint_v

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1))
            pred = self.common_fc_2(aggr_feat)

        if self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], return_all=True)
            pred = self.common_fc_2(aggr_feat)

        return pred, pred_aa, pred_vv, aggr_feat


    def forward(self, x, **kwargs):

        a, v, joint_a, joint_v = self._get_features(x, **kwargs)

        pred_a = a["preds"]["combined"]
        pred_v = v["preds"]["combined"]

        pred, pred_joint_a, pred_joint_v, z12 = self._forward_main(joint_a, joint_v, joint_a["preds"]["combined"], joint_v["preds"]["combined"], **kwargs)

        output = {  "preds":{"combined":pred,
                             "joint_c": pred_joint_a,
                             "joint_g": pred_joint_v,
                            "c":pred_a,
                            "g":pred_v
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"],
                                "joint_c": joint_a["features"]["combined"],
                                "joint_g": joint_v["features"]["combined"],
                                 "z12": z12
                                 }
                  }

        if self.training:
            self.fc_yz1z2.train()
            self.fc_z1z2.train()

            if self.cls_type == "mlp":
                # label = kwargs["label"]
                # one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
                this_pred = pred.detach()
                z12_yz1z2 = self.fc_yz1z2(
                    torch.cat([this_pred, a["features"]["combined"], v["features"]["combined"]], dim=1).detach())
                # z12_yz1z2 = self.fc_yz1z2(torch.cat([one_hot_label, a["features"]["combined"], v["features"]["combined"]], dim=1).detach())
                z12_z1z2 = self.fc_z1z2(
                    torch.cat([a["features"]["combined"], v["features"]["combined"]], dim=1).detach())

            elif self.cls_type == "conformer":
                # label = kwargs["label"]
                # one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
                this_pred = pred.detach()
                this_pred = self.lin_pred(this_pred).unsqueeze(dim=2)
                _, z12_yz1z2, _ = self.fc_yz1z2(
                    torch.cat([this_pred, a["nonaggr_features"]["combined"].detach(), v["nonaggr_features"]["combined"].detach()], dim=2), return_all=True)
                _, z12_z1z2, _ = self.fc_z1z2(
                    torch.cat([a["nonaggr_features"]["combined"].detach(), v["nonaggr_features"]["combined"].detach()], dim=2), return_all=True)
            # cos_sim = nn.CosineSimilarity(di m=1)

            # recon_yz1z2 = 1.0 - cos_sim(z12_yz1z2, z12.detach()).mean()
            # recon_z1z2 = 1.0 - cos_sim(z12_z1z2, z12.detach()).mean()

            recon_yz1z2 = F.mse_loss(z12.detach(), z12_yz1z2)
            recon_z1z2 = F.mse_loss(z12.detach(), z12_z1z2)

            if not hasattr(self, 'optimizer_predheads'):
                self.optimizer_predheads = torch.optim.Adam(
                    list(self.fc_yz1z2.parameters()) + list(self.fc_z1z2.parameters()),
                    lr=self.args.get("lr_predheads", 0.0001)
                )
            self.optimizer_predheads.zero_grad()
            # print("Recon yz1z2: ", recon_yz1z2, "Recon z1z2: ", recon_z1z2)
            ( recon_yz1z2 + recon_z1z2).backward()
            self.optimizer_predheads.step()

            self.fc_yz1z2.eval()
            self.fc_z1z2.eval()

            # for param in self.fc_yz1z2.parameters():
            #     param.requires_grad = False
            # for param in self.fc_z1z2.parameters():
            #     param.requires_grad = False

            if self.cls_type == "mlp":
                # label = kwargs["label"]
                # one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
                this_pred = pred.detach()
                z12_yz1z2 = self.fc_yz1z2(
                    torch.cat([this_pred, a["features"]["combined"], v["features"]["combined"]], dim=1)).detach()
                # z12_yz1z2 = self.fc_yz1z2(torch.cat([one_hot_label, a["features"]["combined"], v["features"]["combined"]], dim=1).detach())
                z12_z1z2 = self.fc_z1z2(
                    torch.cat([a["features"]["combined"], v["features"]["combined"]], dim=1)).detach()

            elif self.cls_type == "conformer":
                # label = kwargs["label"]
                # one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
                this_pred = pred.detach()
                this_pred = self.lin_pred(this_pred).unsqueeze(dim=2)
                _, z12_yz1z2, _ = self.fc_yz1z2(
                    torch.cat([this_pred, a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], dim=2), return_all=True)
                _, z12_z1z2, _ = self.fc_z1z2(
                    torch.cat([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], dim=2), return_all=True)


            bias_infusion = self.args.get("bias_infusion",{})

            recon_yz1z2 = F.mse_loss(z12, z12_yz1z2.detach()) * bias_infusion.get("weight1", 0)
            recon_z1z2 = F.mse_loss(z12, z12_z1z2.detach()) * bias_infusion.get("weight2", 0) * bias_infusion.get("weight1", 0)

            output["losses"]={"sr_yz1z2": recon_yz1z2,
                            "sr_z1z2": - recon_z1z2}

            output["features"]["z12_yz1z2"] = z12_yz1z2
            output["features"]["z12_z1z2"] = z12_z1z2

        return output

class Fusion_Synprom_onlyq(nn.Module):
    def __init__(self, args, encs):
        super(Fusion_Synprom_onlyq, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)

        self.count_trainingsteps = 0

        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(d_model, num_classes)
            )

            self.fc_z1z2 = nn.Sequential(
                nn.Linear(d_model * 2, d_model, bias=False),
            )
            self.optimizer_predheads = torch.optim.Adam(list(self.fc_z1z2.parameters()),
                lr=self.args.get("lr_predheads", 0.0001)
            )
        else:
            raise ValueError("Unknown cls_type")


        # self.fc_1_mu = nn.Linear(d_model,  self.latent_dim, bias=False)
        # self.fc_0_logvar = nn.Linear(d_model, self.latent_dim, bias=False)
        # self.fc_1_logvar = nn.Linear(d_model, self.latent_dim, bias=False)


    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)

        joint_a = self.enc_2(x, **kwargs)
        joint_v = self.enc_3(x, **kwargs)

        return a, v, joint_a, joint_v

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1))
            pred = self.common_fc_2(aggr_feat)

        if self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], return_all=True)
            pred = self.common_fc_2(aggr_feat)

        return pred, pred_aa, pred_vv, aggr_feat


    def forward(self, x, **kwargs):

        a, v, joint_a, joint_v = self._get_features(x, **kwargs)

        pred_a = a["preds"]["combined"]
        pred_v = v["preds"]["combined"]

        pred, pred_joint_a, pred_joint_v, z12 = self._forward_main(joint_a, joint_v, joint_a["preds"]["combined"], joint_v["preds"]["combined"], **kwargs)

        output = {  "preds":{"combined":pred,
                             "joint_c": pred_joint_a,
                             "joint_g": pred_joint_v,
                            "c":pred_a,
                            "g":pred_v
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"],
                                "joint_c": joint_a["features"]["combined"],
                                "joint_g": joint_v["features"]["combined"],
                                 "z12": z12
                                 }
                  }

        if self.training:
            z1 = a["features"]["combined"]
            z2 = v["features"]["combined"]
            z1 = l2_normalize(z1)
            z2 = l2_normalize(z2)
            z12 = l2_normalize(z12)

            # --- Step 1: Train q (adversary) ---
            self.fc_z1z2.train()
            joint_feats = torch.cat([z1, z2], dim=1).detach()

            for _ in range(1):  # k=5
                z12_hat_q = self.fc_z1z2(joint_feats)
                z12_hat_q = l2_normalize(z12_hat_q)
                loss_q = -cosine_clamp_loss(z12.detach(), z12_hat_q).mean()
                self.optimizer_predheads.zero_grad()
                loss_q.backward()
                self.optimizer_predheads.step()

            self.fc_z1z2.eval()

            # --- Step 2: Compute unpredictability loss for fusion ---
            z12_hat_q = self.fc_z1z2(joint_feats).detach()
            z12_hat_q = l2_normalize(z12_hat_q)
            loss_unpred = cosine_clamp_loss(z12, z12_hat_q)

            # --- Hard-example weighting (intersection of unimodal errors) ---
            label = kwargs["label"]
            with torch.no_grad():
                y1 = pred_a.argmax(-1)
                y2 = pred_v.argmax(-1)
                mask = ((y1 != label) & (y2 != label)).float()
                weights = 0.9 * mask + 0.1  # α=0.8

            loss_unpred = (loss_unpred * weights).mean()

            # --- Store losses ---
            bias_infusion = self.args.get("bias_infusion", {})
            output["losses"] = {
                "sr_z1z2": bias_infusion.get("weight2", 1.0) * loss_unpred
            }

            # output["features"]["z12_yz1z2"] = z12_yz1z2
            # output["features"]["z12_z1z2"] = z12_z1z2

        return output


class Fusion_Synprom_unireg(nn.Module):
    def __init__(self, args, encs):
        super(Fusion_Synprom_unireg, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)

        self.count_trainingsteps = 0
        bias_infusion = args.get("bias_infusion",{})
        self.weight1 = bias_infusion.get("weight1", 0)
        self.weight2 = bias_infusion.get("weight2", 0)

        distance_metric = bias_infusion.get("dmetric", "mse")
        if distance_metric == "mse":
            self.distance_metric = F.mse_loss
            print("Using MSE distance metric")
        elif distance_metric == "cosine":
            self.distance_metric = nn.CosineSimilarity()
            print("Using Cosine similarity distance metric")



        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            if self.weight1 !=0:
                self.proj_z12 = nn.Linear(d_model, 32)
                self.fc_yz2 = nn.Sequential(
                    nn.Linear(num_classes + d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 32),
                )
                self.fc_yz1 = nn.Sequential(
                    nn.Linear(num_classes + d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 32),
                )

                self.fc_z1 = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 32),
                )
                self.fc_z2 = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 32),
                )

        elif self.cls_type == "conformer":

            self.common_fc_1 =TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6, output_dim=d_model)

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            if self.weight1 !=0:
                self.lin_pred = nn.Linear(num_classes, d_model)
                self.fc_yz2 = TF_Proc(
                    input_dim=d_model,
                    dim=d_model,
                    layers=2, output_dim=d_model)

                self.fc_yz1 = TF_Proc(
                    input_dim=d_model,
                    dim=d_model,
                    layers=2, output_dim=d_model)

                self.fc_z1 = TF_Proc(
                    input_dim=d_model,
                    dim=d_model,
                    layers=2, output_dim=d_model)

                self.fc_z2 = TF_Proc(
                    input_dim=d_model,
                    dim=d_model,
                    layers=2, output_dim=d_model)

        else:
            raise ValueError("Unknown cls_type")



    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)

        joint_a = self.enc_2(x, **kwargs)
        joint_v = self.enc_3(x, **kwargs)

        return a, v, joint_a, joint_v

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1))
            pred = self.common_fc_2(aggr_feat)

        if self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], return_all=True)
            pred = self.common_fc_2(aggr_feat)

        return pred, pred_aa, pred_vv, aggr_feat


    def forward(self, x, **kwargs):

        a, v, joint_a, joint_v = self._get_features(x, **kwargs)

        pred_a = a["preds"]["combined"]
        pred_v = v["preds"]["combined"]

        pred, pred_joint_a, pred_joint_v, z12 = self._forward_main(joint_a, joint_v, joint_a["preds"]["combined"], joint_v["preds"]["combined"], **kwargs)

        output = {  "preds":{   "combined":pred,
                                "joint_c": pred_joint_a,
                                "joint_g": pred_joint_v,
                                "c":pred_a,
                                "g":pred_v
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"],
                                "joint_c": joint_a["features"]["combined"],
                                "joint_g": joint_v["features"]["combined"],
                                 }
                  }

        if self.training:


            self.fc_yz1.train()
            self.fc_yz2.train()
            self.fc_z1.train()
            self.fc_z2.train()

            if self.cls_type == "mlp":
                # label = kwargs["label"]
                # one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
                # this_pred = one_hot_label.detach()
                this_pred = pred.detach()
                z12_yz1 = self.fc_yz1( torch.cat([this_pred, a["features"]["combined"]], dim=1).detach())
                z12_yz2 = self.fc_yz2( torch.cat([this_pred, v["features"]["combined"]], dim=1).detach())
                z12_z1 = self.fc_z1(a["features"]["combined"].detach())
                z12_z2 = self.fc_z2(v["features"]["combined"].detach())
                proj_z12 = self.proj_z12(z12.detach())

            elif self.cls_type == "conformer":
                # label = kwargs["label"]
                # one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
                # this_pred = one_hot_label.detach()
                this_pred = pred.detach()
                this_pred = self.lin_pred(this_pred).unsqueeze(dim=2)
                z12_yz1 = self.fc_yz1(torch.cat([this_pred, a["nonaggr_features"]["combined"]], dim=2).detach())
                z12_yz2 = self.fc_yz2(torch.cat([this_pred, v["nonaggr_features"]["combined"]], dim=2).detach())
                z12_z1 = self.fc_z1(a["nonaggr_features"]["combined"].detach())
                z12_z2 = self.fc_z2(v["nonaggr_features"]["combined"].detach())


            recon_yz1 = self.distance_metric(proj_z12, z12_yz1)
            recon_yz2 = self.distance_metric(proj_z12, z12_yz2)
            recon_z1 = self.distance_metric(proj_z12, z12_z1)
            recon_z2 = self.distance_metric(proj_z12, z12_z2)

            if not hasattr(self, 'optimizer_predheads'):
                self.optimizer_predheads = torch.optim.Adam(
                    list(self.fc_yz2.parameters()) + list(self.fc_yz1.parameters()) + list(self.fc_z1.parameters()) + list(self.fc_z2.parameters()),
                    lr=self.args.get("lr_predheads", 0.0001)
                )
            self.optimizer_predheads.zero_grad()
            if len(recon_yz1.shape) > 0:
                recon_yz1 = recon_yz1.mean() + 1
                recon_yz2 = recon_yz2.mean() + 1
                recon_z1 = recon_z1.mean() + 1
                recon_z2 = recon_z2.mean() + 1
                ( recon_yz1 + recon_yz2 + recon_z1 + recon_z2).backward()
            else:
                ( recon_yz1 + recon_yz2 + recon_z1 + recon_z2).backward()
            self.optimizer_predheads.step()

            self.fc_yz1.eval()
            self.fc_yz2.eval()
            self.fc_z1.eval()
            self.fc_z2.eval()

            # for param in self.fc_yz1z2.parameters():
            #     param.requires_grad = False
            # for param in self.fc_z1z2.parameters():
            #     param.requires_grad = False

            if self.cls_type == "mlp":
                # label = kwargs["label"]
                # one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
                # this_pred = one_hot_label.detach()
                this_pred = pred.detach()
                z12_yz1 = self.fc_yz1( torch.cat([this_pred, a["features"]["combined"]], dim=1).detach())
                z12_yz2 = self.fc_yz2( torch.cat([this_pred, v["features"]["combined"]], dim=1).detach())
                z12_z1 = self.fc_z1(a["features"]["combined"].detach())
                z12_z2 = self.fc_z2(v["features"]["combined"].detach())
                proj_z12 = self.proj_z12(z12)


            elif self.cls_type == "conformer":
                # label = kwargs["label"]
                # one_hot_label = F.one_hot(label, num_classes=self.num_classes).float()
                # this_pred = one_hot_label.detach()
                this_pred = pred.detach()
                this_pred = self.lin_pred(this_pred).unsqueeze(dim=2)
                z12_yz1 = self.fc_yz1(torch.cat([this_pred, a["nonaggr_features"]["combined"]], dim=2).detach())
                z12_yz2 = self.fc_yz2(torch.cat([this_pred, v["nonaggr_features"]["combined"]], dim=2).detach())
                z12_z1 = self.fc_z1(a["nonaggr_features"]["combined"].detach())
                z12_z2 = self.fc_z2(v["nonaggr_features"]["combined"].detach())


            bias_infusion = self.args.get("bias_infusion",{})

            recon_yz1z2 = self.distance_metric(proj_z12, 0.5*(z12_yz1.detach() + z12_yz2.detach())) * bias_infusion.get("weight1", 0)
            recon_z1z2 = self.distance_metric(proj_z12, 0.5*(z12_z1.detach() + z12_z2.detach())) * bias_infusion.get("weight2", 0) * bias_infusion.get("weight1", 0)
            if len(recon_yz1z2.shape) > 0:
                recon_yz1z2 = recon_yz1z2.mean() + 1
                recon_z1z2 = recon_z1z2.mean() + 1
            output["losses"]={
                "sr_yz1z2": recon_yz1z2,
                "sr_z1z2": - recon_z1z2
            }


        return output

class Fusion_Synprom_perf(nn.Module):
    def __init__(self, args, encs):
        super(Fusion_Synprom_perf, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)

        self.count_trainingsteps = 0
        bias_infusion = args.get("bias_infusion",{})
        self.l = bias_infusion.get("l", 0)
        self.weight2 = bias_infusion.get("weight2", 0)

        distance_metric = bias_infusion.get("dmetric", "mse")
        if distance_metric == "mse":
            self.distance_metric = F.mse_loss
            print("Using MSE distance metric")
        elif distance_metric == "cosine":
            self.distance_metric = nn.CosineSimilarity()
            print("Using Cosine similarity distance metric")



        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "conformer":

            self.common_fc_1 =TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6, output_dim=d_model)

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
            if self.weight1 !=0:
                self.lin_pred = nn.Linear(num_classes, d_model)
                self.fc_yz2 = TF_Proc(
                    input_dim=d_model,
                    dim=d_model,
                    layers=2, output_dim=d_model)

                self.fc_yz1 = TF_Proc(
                    input_dim=d_model,
                    dim=d_model,
                    layers=2, output_dim=d_model)

                self.fc_z1 = TF_Proc(
                    input_dim=d_model,
                    dim=d_model,
                    layers=2, output_dim=d_model)

                self.fc_z2 = TF_Proc(
                    input_dim=d_model,
                    dim=d_model,
                    layers=2, output_dim=d_model)

        else:
            raise ValueError("Unknown cls_type")



    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)

        joint_a = self.enc_2(x, **kwargs)
        joint_v = self.enc_3(x, **kwargs)

        return a, v, joint_a, joint_v

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1))
            pred = self.common_fc_2(aggr_feat)

        if self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], return_all=True)
            pred = self.common_fc_2(aggr_feat)

        return pred, pred_aa, pred_vv, aggr_feat


    def forward(self, x, **kwargs):

        a, v, joint_a, joint_v = self._get_features(x, **kwargs)

        pred_a = a["preds"]["combined"]
        pred_v = v["preds"]["combined"]

        pred, pred_joint_a, pred_joint_v, z12 = self._forward_main(joint_a, joint_v, joint_a["preds"]["combined"], joint_v["preds"]["combined"], **kwargs)

        output = {  "preds":{   "combined":pred,
                                "joint_c": pred_joint_a,
                                "joint_g": pred_joint_v,
                                "c":pred_a,
                                "g":pred_v
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"],
                                "joint_c": joint_a["features"]["combined"],
                                "joint_g": joint_v["features"]["combined"],
                                 }
                  }

        if self.training:
            ground_truth_labels = kwargs["label"]

            #softmax

            p_multi = F.softmax(pred, dim=1).gather(1, ground_truth_labels.view(-1, 1))
            p_uni1 = F.softmax(pred_a, dim=1).gather(1, ground_truth_labels.view(-1, 1)).detach()
            p_uni2 = F.softmax(pred_v, dim=1).gather(1, ground_truth_labels.view(-1, 1)).detach()

            margin = 0

            loss_synergy1 = torch.mean(p_uni1 - p_multi + margin)+1
            loss_synergy2 = torch.mean(p_uni2 - p_multi + margin)+1

            wandb_loss = {"reg_loss": {"loss_synergy1": loss_synergy1, "loss_synergy2": loss_synergy2}}
            wandb.log(wandb_loss, step=kwargs["current_step"]+1)

            synergy_loss = loss_synergy1 + loss_synergy2
            output["losses"]={
                "sl": synergy_loss*self.l,
            }


        return output

class Z1Generator(nn.Module):
    """
    Conditional encoder-decoder that maps Z2 -> tilde_Z1 -> reconstructed Z1,
    or symmetrically Z1 -> tilde_Z2 -> reconstructed Z2.
    """
    def __init__(self, dim_in, dim_out, hidden_dim=128, bottleneck_dim=64, variational=False):
        super().__init__()
        self.variational = variational

        # Encoder: input modality -> bottleneck
        self.enc = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if variational:
            self.enc_mu = nn.Linear(hidden_dim, bottleneck_dim)
            self.enc_logvar = nn.Linear(hidden_dim, bottleneck_dim)
        else:
            self.enc_out = nn.Linear(hidden_dim, bottleneck_dim)

        # Decoder: bottleneck -> reconstruct opposite modality
        self.dec = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, z_in, z_target=None, add_noise=False, **kwargs):

        h = self.enc(z_in.detach())

        if self.variational:
            mu = self.enc_mu(h)
            logvar = self.enc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            interim_z = mu + eps * std
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            interim_z = self.enc_out(h)
            kl = torch.tensor(0.0, device=z_in.device)

        z_hat = self.dec(interim_z)

        losses = {}
        if z_target is not None:
            rec = F.mse_loss(z_hat, z_target)
            losses = {"rec": rec, "kl": kl}

        if add_noise:
            z_hat = z_hat + torch.randn_like(z_hat) * 0.1

        return {"interim_z": interim_z, "tilde": z_hat, "losses": losses}

class Z1Generator_v2(nn.Module):
    """
    Conditional encoder-decoder that maps Z2 -> tilde_Z1,
    or symmetrically Z1 -> tilde_Z2.
    """
    def __init__(self, args, encs, **kwargs):
        super().__init__()

        dim_in = args.get("dim_in", 128)
        dim_out = args.get("dim_out", 128)
        hidden_dim = args.get("hidden_dim", 128)
        bottleneck_dim = args.get("bottleneck_dim", 64)
        variational = args.get("variational", False)

        self.args = args
        self.encs = encs
        self.variational = variational

        # Encoder: input modality -> bottleneck
        self.enc = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if variational:
            self.enc_mu = nn.Linear(hidden_dim, bottleneck_dim)
            self.enc_logvar = nn.Linear(hidden_dim, bottleneck_dim)
        else:
            self.enc_out = nn.Linear(hidden_dim, bottleneck_dim)

        # Decoder: bottleneck -> reconstruct opposite modality
        self.dec = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, z_in, z_target=None, add_noise=False):

        h = self.enc(z_in.detach())

        if self.variational:
            mu = self.enc_mu(h)
            logvar = self.enc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            interim_z = mu + eps * std
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            interim_z = self.enc_out(h)
            kl = torch.tensor(0.0, device=z_in.device)

        z_hat = self.dec(interim_z)

        losses = {}
        if z_target is not None:
            rec = F.mse_loss(z_hat, z_target)
            # pseudo_rec_diff = F.mse_loss(z_in, z_target).detach()
            # perm = torch.randperm(z_in.size(0))
            # pseudo_rec_same = F.mse_loss(z_in, z_in[perm]).detach()
            # pseudo_rec_same = F.mse_loss(z_target, z_target).detach()

            # losses = {"rec": rec, "kl": kl, "prec_diff":pseudo_rec_diff, "prec_same":pseudo_rec_same}
            losses = {"rec": rec, "kl": kl}

        if add_noise:
            z_hat = z_hat + torch.randn_like(z_hat) * 0.1

        return {"interim_z": interim_z, "tilde": z_hat, "losses": losses}

class Z1Generator_v3(nn.Module):
    """
    Conditional encoder-decoder that maps Z2 -> tilde_Z1,
    or symmetrically Z1 -> tilde_Z2.
    """
    def __init__(self, args, encs, **kwargs):
        super().__init__()

        dim_in = args.get("dim_in", 128)
        dim_out = args.get("dim_out", 128)
        hidden_dim = args.get("hidden_dim", 128)
        bottleneck_dim = args.get("bottleneck_dim", 64)
        variational = args.get("variational", False)

        self.args = args
        self.encs = encs
        self.variational = variational

        # Encoder: input modality -> bottleneck
        self.vaeenc = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if variational:
            self.vaeenc_mu = nn.Linear(hidden_dim, bottleneck_dim)
            self.vaeenc_logvar = nn.Linear(hidden_dim, bottleneck_dim)
        else:
            self.vaeenc_out = nn.Linear(hidden_dim, bottleneck_dim)

        # Decoder: bottleneck -> reconstruct opposite modality
        self.vaedec = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, z_in, z_target=None, add_noise=False, **kwargs):

        h = self.vaeenc(z_in.detach())

        if self.variational:
            mu = self.vaeenc_mu(h)
            logvar = self.vaeenc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            interim_z = mu + eps * std
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            interim_z = self.vaeenc_out(h)
            kl = torch.tensor(0.0, device=z_in.device)

        z_hat = self.vaedec(interim_z)

        losses = {}
        if z_target is not None:
            rec = F.mse_loss(z_hat, z_target)
            # pseudo_rec_diff = F.mse_loss(z_in, z_target).detach()
            # perm = torch.randperm(z_in.size(0))
            # pseudo_rec_same = F.mse_loss(z_in, z_in[perm]).detach()
            # pseudo_rec_same = F.mse_loss(z_target, z_target).detach()

            # losses = {"rec": rec, "kl": kl, "prec_diff":pseudo_rec_diff, "prec_same":pseudo_rec_same}
            losses = {"rec": rec, "kl": kl}

        if add_noise:
            z_hat = z_hat + torch.randn_like(z_hat) * 0.1

        return {"interim_z": interim_z, "tilde": z_hat, "losses": losses}


class Fusion_Synprom_IB_Input(nn.Module):
    def __init__(self, args, encs):
        super(Fusion_Synprom_IB_Input, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        self.perturb = {"type": args.perturb}


        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        bias_infusion = args.get("bias_infusion",{})
        self.l = bias_infusion.get("l", 0)
        self.contrcoeff = bias_infusion.get("contrcoeff", False)



        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )
            self.mu_head = nn.Linear(d_model, num_classes)
            self.logvar_head = nn.Linear(d_model, num_classes)

        elif self.cls_type == "conformer":

            self.common_fc_1 =TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6, output_dim=d_model)

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )

        else:
            raise ValueError("Unknown cls_type")

        self.mu_head = nn.Linear(fc_inner, num_classes)
        self.logvar_head = nn.Linear(fc_inner, num_classes)

        # self.z1generator = Z1Generator_v2(d_model, d_model, hidden_dim=128, bottleneck_dim=128, variational=False)
        # self.z2generator = Z1Generator_v2(d_model, d_model, hidden_dim=128, bottleneck_dim=128, variational=False)

        self.bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.image_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)

        return a, v

    def _get_features_pert(self, x, **kwargs):
        """
        x: the usual batch dict (already preprocessed)
           x["data"][2] is the image tensor [B,3,H,W]

        enc_2: caption generator (image → text)
        enc_3: image generator  (image → image)
        enc_0: text encoder
        enc_1: image encoder

        Returns:
            a = perturbed text features
            v = perturbed image features
        """

        generated_captions = self.enc_2(x, **kwargs)  # list of strings (len B)
        generated_images = self.enc_3(x, **kwargs)  # list of PIL images or tensors


        bert_tok = self.bert_tokenizer(
            generated_captions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).cuda()

        pseudo_text = {
            "input_ids": bert_tok["input_ids"],
            "attention_mask": bert_tok["attention_mask"]
        }


        processed_images = []

        for img in generated_images:
            # Convert tensor → PIL if needed
            if isinstance(img, torch.Tensor):
                # ensure valid range
                if img.max() > 1:
                    img = img / 255.0
                img = img.clamp(0, 1)
                img = to_pil_image(img.cpu())
            processed_images.append(self.image_tf(img))

        pseudo_image = torch.stack(processed_images, dim=0).to(device)  # [B,3,224,224]

        out_text = self.enc_0({0: pseudo_text}, **kwargs)
        out_image = self.enc_1({2: pseudo_image}, **kwargs)

        a = out_text["features"]["combined"]
        v = out_image["features"]["combined"]

        return a, v

    def _forward_main(self, a, v, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1))
            feat = self.common_fc_2(aggr_feat)

        if self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], return_all=True)
            feat = self.common_fc_2(aggr_feat)

        pred_mu = self.mu_head(feat)
        pred_logvar = self.logvar_head(feat)
        if "stochastic" in kwargs and kwargs["stochastic"]:
            std = torch.exp(0.5 * pred_logvar)
            eps = torch.randn_like(std)
            pred = pred_mu + eps * std

            return pred, pred_mu, pred_logvar
        # else:
        #     std = torch.exp(0.5 * pred_logvar)
        #     eps = torch.randn_like(std)
        #     pred_mu = pred_mu + eps * std

        return pred_mu, aggr_feat

    def forward_once(self, x, **kwargs):

        a, v = self._get_features(x, **kwargs)

        if "perturb_x1" in kwargs and kwargs["perturb_x1"]:
            pert_out = self._perturb_features(x, z1=a["features"]["combined"], z2=v["features"]["combined"], swap_direction=True, **kwargs)
            a["features"]["combined"] = pert_out["tilde"].detach()
            # v["features"]["combined"] = v["features"]["combined"].detach()
            pert_losses = {"px1":pert_out["losses"]}

        if "perturb_x2" in kwargs and kwargs["perturb_x2"]:
            pert_out = self._perturb_features(x, z1=v["features"]["combined"], z2=a["features"]["combined"], **kwargs)
            v["features"]["combined"] = pert_out["tilde"].detach()
            # a["features"]["combined"] = a["features"]["combined"].detach()
            pert_losses = {"px2":pert_out["losses"]}



        if "stochastic" in kwargs and kwargs["stochastic"]:
            pred, mu, logvar = self._forward_main(a, v, **kwargs)
            pl = pert_losses if "pert_losses" in locals() else {}
            return self._kl_loss(mu, logvar), pl

        pred, feat = self._forward_main(a, v, **kwargs)

        output = {  "preds":{   "combined":pred,
                                "p1":a["preds"]["combined"],
                                "p2":v["preds"]["combined"]
                            },
                    "features": {"z1": a["features"]["combined"],
                                "z2": v["features"]["combined"]
                                 },
                    "losses": {}
                  }

        return output

    def _perturb_features(self, x, z1, z2, **kwargs):

        if self.perturb["type"] == "zeros":
            return {"tilde":torch.zeros_like(z1).detach()}
        elif self.perturb["type"] == "cmn": #conditional masked and noisy
            mask = (torch.rand_like(z1) < self.perturb.get("p",0.5)).float()  # partial mask
            scale = self.perturb.get("s", 0.1) * (1 + torch.tanh(z2.norm(dim=-1, keepdim=True) / self.perturb.get("c",1)))
            noise = torch.randn_like(z1) * scale
            tilde_z1 = mask * z1 + (1 - mask) * noise
            return {"tilde": tilde_z1.detach()}
        elif self.perturb["type"] == "gen" or  self.perturb["type"] == "gen_v2":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, current_step=kwargs.get("current_step", None))
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, current_step=kwargs.get("current_step", None))
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out
        elif self.perturb["type"] == "gennoise":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out

        elif self.perturb["type"] == "geninput":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(x, compute_loss=self.training, current_step=kwargs.get("current_step", None))
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(x, compute_loss=self.training, current_step=kwargs.get("current_step", None))
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out
        # noise = torch.randn_like(feat) * 0.1
        # return feat + noise

    def _kl_loss(self, mu_p, logvar_p):
        return 0.5 * torch.sum( torch.exp(logvar_p) + mu_p**2 - 1.0 - logvar_p, dim=1 ).mean()

    def forward(self, x, **kwargs):

        output = self.forward_once(x, **kwargs)

        if self.training:
            output_perturbations = self._get_features_pert(x, **kwargs)

            kl_1, pert_losses = self.forward_once( x, stochastic=True, perturb_x1=True, perturb_x2=False, **kwargs)
            output["losses"].update(pert_losses)

            kl_2, pert_losses = self.forward_once(x, stochastic=True, perturb_x1=False, perturb_x2=True, **kwargs)
            output["losses"].update(pert_losses)


            infonce_loss = nt_xent_loss(output["features"]["z1"], output["features"]["z2"], temperature=1.0)

            # supcon_loss = nt_xent_loss(output["features"]["c"], output["features"]["g"], label=label, temperature=self.temperature)

            wandb_loss = {"reg_loss": {"kl_1": kl_1, "kl_2": kl_2, "infonce": infonce_loss}}
            wandb.log(wandb_loss, step=kwargs["current_step"]+1)

            output["losses"].update({
                "sl_1": kl_1*self.l,
                "sl_2": kl_2*self.l,
                "infonce": infonce_loss*self.contrcoeff
            })
        else:
            output_z1 = self.forward_once( x, stochastic=False, perturb_x1=True, perturb_x2=False, **kwargs)
            output_z2 = self.forward_once( x, stochastic=False, perturb_x1=False, perturb_x2=True, **kwargs)
            output["features"]["~z1"] = output_z1["features"]["z1"]
            output["features"]["~z2"] = output_z2["features"]["z2"]


        return output


class Z1Generator_GAN(nn.Module):
    """
    Conditional GAN version of Z1Generator_v3.

    Features:
      - Adversarial modes: 'bce' | 'hinge' | 'wgan'
      - Spectral norm (D), instance noise, label smoothing (BCE)
      - Feature matching (on D's penultimate features)
      - Gradient penalty (WGAN-GP)
      - Optional safe clamp on D scores

    Args
    ----
    args : Dict[str, Any]
        Hyperparameters & toggles, e.g.:
          dim_in:int, dim_out:int, hidden_dim:int, bottleneck_dim:int, noise_dim:int,
          lambda_rec:float, lambda_d:float, gan_type:str, lambda_gp:float,
          feature_matching_weight:float, spectral_norm:bool, d_dropout:float,
          input_noise_std:float, bce_label_smooth:float, safe_clamp:Optional[float]
    """

    # ---------- attribute annotations (for IDEs & static checkers) ----------
    args: Dict[str, Any]
    dim_in: int
    dim_out: int
    hidden_dim: int
    bottleneck_dim: int
    noise_dim: int
    warmup_steps: int
    lambda_rec: float
    lambda_d: float
    gan_type: str
    lambda_gp: float
    fm_weight: float
    use_spectral_norm: bool
    d_dropout: float
    input_noise_std: float
    bce_label_smooth: float
    safe_clamp: Optional[float]
    gen_enc: nn.Sequential
    gen_dec: nn.Sequential
    disc_feat: nn.Sequential
    disc_out: nn.Linear

    def __init__(self, args: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__()

        # ---- Config ----
        self.args = args
        self.dim_in = int(args.get("dim_in", 128))
        self.dim_out = int(args.get("dim_out", 128))
        self.hidden_dim = int(args.get("hidden_dim", 128))
        self.bottleneck_dim = int(args.get("bottleneck_dim", 64))
        self.noise_dim = int(args.get("noise_dim", 32))

        # loss weights & modes
        self.lambda_rec = float(args.get("lambda_rec", 0.5))
        self.lambda_d = float(args.get("lambda_d", 0.3))
        self.gan_type = str(args.get("gan_type", "hinge")).lower()
        self.lambda_gp = float(args.get("lambda_gp", 25.0))
        self.fm_weight = float(args.get("feature_matching_weight", 0.1))

        # regularizers
        self.use_spectral_norm = bool(args.get("spectral_norm", True))
        self.d_dropout = float(args.get("d_dropout", 0.0))
        self.input_noise_std = float(args.get("input_noise_std", 0.05))
        self.bce_label_smooth = float(args.get("bce_label_smooth", 0.1))
        self.safe_clamp = args.get("safe_clamp", 10.0)  # Optional[float] or None

        self.warmup_steps = int(args.get("warmup_steps", 1000))

        # ---- Generator ----
        self.gen_enc = nn.Sequential(
            nn.Linear(self.dim_in + self.noise_dim, self.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.ReLU(),
        )
        self.gen_dec = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim * 2, self.dim_out),
        )
        # ---- Discriminator ----
        def maybe_sn(layer: nn.Module) -> nn.Module:
            return SN(layer) if self.use_spectral_norm else layer

        d_layers = [
            maybe_sn(nn.Linear(self.dim_in + self.dim_out, self.hidden_dim)),
            nn.ReLU(),
        ]
        if self.d_dropout > 0:
            d_layers.append(nn.Dropout(self.d_dropout))
        d_layers += [
            maybe_sn(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        ]
        if self.d_dropout > 0:
            d_layers.append(nn.Dropout(self.d_dropout))

        self.disc_feat = nn.Sequential(*d_layers)
        self.disc_out = maybe_sn(nn.Linear(self.hidden_dim, 1))

    # --------------------------------------------------------
    # Core functions
    # --------------------------------------------------------
    def generate(self, z_in: torch.Tensor) -> torch.Tensor:
        """
        Generate tilde-Z (fake target) conditioned on z_in.

        Parameters
        ----------
        z_in : torch.Tensor  [B, dim_in]

        Returns
        -------
        torch.Tensor  [B, dim_out]
        """
        noise = torch.randn(z_in.size(0), self.noise_dim, device=z_in.device)
        return self.gen_dec(self.gen_enc(torch.cat([z_in, noise], dim=-1)))

    def _add_instance_noise(self, z: torch.Tensor) -> torch.Tensor:
        """
        Optionally add Gaussian instance noise to stabilize D.

        Parameters
        ----------
        z : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        if self.input_noise_std > 0:
            z = z + self.input_noise_std * torch.randn_like(z)
        return z

    def discriminate_raw(self, z_in: torch.Tensor, z_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminator forward returning both logits and intermediate features.

        Parameters
        ----------
        z_in : torch.Tensor   [B, dim_in]
        z_target : torch.Tensor [B, dim_out]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            logits: [B, 1], feats: [B, hidden_dim]
        """
        z_target = self._add_instance_noise(z_target)
        x = torch.cat([z_in, z_target], dim=-1)
        feats = self.disc_feat(x)
        logits = self.disc_out(feats)
        return logits, feats

    def discriminate(self, z_in: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Discriminator forward returning only logits (optionally clamped).

        Parameters
        ----------
        z_in : torch.Tensor
        z_target : torch.Tensor

        Returns
        -------
        torch.Tensor  [B, 1]
        """
        logits, _ = self.discriminate_raw(z_in, z_target)
        if self.safe_clamp is not None:
            logits = torch.clamp(logits, -float(self.safe_clamp), float(self.safe_clamp))
        return logits

    # --------------------------------------------------------
    # Gradient penalty (for WGAN-GP)
    # --------------------------------------------------------
    def gradient_penalty(self, z_in: torch.Tensor, z_target: torch.Tensor, z_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute WGAN-GP gradient penalty on interpolates.

        Returns
        -------
        torch.Tensor  [] scalar
        """
        batch = z_in.size(0)
        eps = torch.rand(batch, 1, device=z_in.device).expand_as(z_target)
        interpolates = (eps * z_target + (1 - eps) * z_fake).requires_grad_(True)
        d_interpolates = self.discriminate(z_in, interpolates)
        grads = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.view(batch, -1)
        gp = ((grads.norm(2, dim=1).clamp(max=10.0) - 1.0) ** 2).mean()
        return gp

    # --------------------------------------------------------
    # Compute losses
    # --------------------------------------------------------
    def compute_losses(
            self,
            z_in: torch.Tensor,
            z_target: torch.Tensor,
            current_step: Optional[int] = None
    ):
        """
        Computes losses for both G and D.
        Includes two regimes:
            1) Warm-up phase (no discriminator training)
            2) Full adversarial phase
        """

        # ========================================================
        # 0. Generate fake samples
        # ========================================================
        z_fake = self.generate(z_in)

        # ========================================================
        # 1. WARM-UP PHASE  (Generator-only training)
        # ========================================================
        if current_step is not None and current_step < self.warmup_steps:

            # Feature matching still allowed
            _, real_feats = self.discriminate_raw(z_in, z_target)
            _, fake_feats_g = self.discriminate_raw(z_in, z_fake)

            # Reconstruction loss only
            g_rec = F.mse_loss(z_fake, z_target)

            # Optional feature matching
            if self.fm_weight > 0:
                g_fm = F.l1_loss(real_feats.mean(0), fake_feats_g.mean(0))
            else:
                g_fm = torch.zeros((), device=z_in.device)

            g_loss = self.lambda_rec * g_rec + self.fm_weight * g_fm

            losses = {
                "total": g_loss,  # Only G updates
                "g_loss": g_loss.detach(),
                "d_loss": torch.tensor(0.0, device=z_in.device),
                "g_adv": torch.tensor(0.0, device=z_in.device),
                "g_rec": g_rec.detach(),
                "g_fm": (self.fm_weight * g_fm).detach(),
                "d_real": torch.tensor(0.0, device=z_in.device),
                "d_fake": torch.tensor(0.0, device=z_in.device),
            }

            return losses, z_fake

        # ========================================================
        # 2. FULL ADVERSARIAL PHASE (normal GAN mode)
        # ========================================================

        # ---- Discriminator outputs ----
        real_pred, real_feats = self.discriminate_raw(z_in, z_target)
        fake_pred = self.discriminate(z_in, z_fake.detach())
        fake_pred_g, fake_feats_g = self.discriminate_raw(z_in, z_fake)

        # Optional clamp for stability
        if self.safe_clamp is not None:
            real_pred = torch.clamp(real_pred, -self.safe_clamp, self.safe_clamp)
            fake_pred = torch.clamp(fake_pred, -self.safe_clamp, self.safe_clamp)
            fake_pred_g = torch.clamp(fake_pred_g, -self.safe_clamp, self.safe_clamp)

        gan_type = self.gan_type

        # ========================================================
        # 2A. DISCRIMINATOR LOSSES
        # ========================================================
        if gan_type == "bce":
            smooth = self.bce_label_smooth
            real_labels = torch.ones_like(real_pred) * (1.0 - smooth)
            fake_labels = torch.zeros_like(fake_pred) + smooth

            d_loss_real = F.binary_cross_entropy_with_logits(real_pred, real_labels)
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # Generator adversarial
            g_adv = F.binary_cross_entropy_with_logits(
                fake_pred_g,
                torch.ones_like(fake_pred_g) * (1.0 - smooth)
            )

        elif gan_type == "hinge":
            d_loss_real = F.relu(1.0 - real_pred).mean()
            d_loss_fake = F.relu(1.0 + fake_pred).mean()
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # Generator adversarial
            g_adv = -fake_pred_g.mean()

        elif gan_type == "wgan":
            d_loss = fake_pred.mean() - real_pred.mean()
            gp = self.gradient_penalty(z_in, z_target, z_fake.detach())
            d_loss = d_loss + self.lambda_gp * gp

            # for logging clarity
            d_loss_real = -real_pred.mean()
            d_loss_fake = fake_pred.mean()

            # Generator adversarial
            g_adv = -fake_pred_g.mean()

        else:
            raise ValueError(f"Unknown gan_type '{gan_type}'")

        # ========================================================
        # 2B. GENERATOR AUXILIARY LOSSES
        # ========================================================
        g_rec = F.mse_loss(z_fake, z_target)

        if self.fm_weight > 0:
            g_fm = F.l1_loss(real_feats.mean(0), fake_feats_g.mean(0))
        else:
            g_fm = torch.zeros((), device=z_in.device)

        # ========================================================
        # 2C. TOTAL GENERATOR & TOTAL LOSS
        # ========================================================
        g_loss = g_adv + self.lambda_rec * g_rec + self.fm_weight * g_fm

        total_loss = g_loss + self.lambda_d * d_loss

        # ========================================================
        # 2D. Return detailed losses
        # ========================================================
        losses = {
            "total": total_loss,
            "g_loss": g_loss.detach(),
            "d_loss": d_loss.detach(),
            "g_adv": g_adv.detach(),
            "g_rec": g_rec.detach(),
            "g_fm": (self.fm_weight * g_fm).detach(),
            "d_real": d_loss_real.detach(),
            "d_fake": d_loss_fake.detach(),
        }

        return losses, z_fake

    # --------------------------------------------------------
    # wandb logging (metrics only)
    # --------------------------------------------------------
    def wandb_log(
        self,
        z_in: torch.Tensor,
        z_target: torch.Tensor,
        z_fake: torch.Tensor,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        fake_pred_g: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """
        Log interpretable GAN diagnostics to wandb (no raw losses).

        Returns
        -------
        Dict[str, float] : metrics dict that was logged
        """
        D_conf_real = torch.sigmoid(real_pred).mean().item()
        D_conf_fake = torch.sigmoid(fake_pred).mean().item()
        D_conf_fake_Gview = torch.sigmoid(fake_pred_g).mean().item()

        with torch.no_grad():
            mean_real = z_target.mean(dim=0)
            mean_fake = z_fake.mean(dim=0)
            cosine_sim = F.cosine_similarity(mean_real, mean_fake, dim=0).item()
            std_diff = (z_target.std(dim=0) - z_fake.std(dim=0)).abs().mean().item()

        adv_balance = D_conf_real - D_conf_fake
        probs = torch.sigmoid(torch.cat([real_pred, fake_pred], dim=0))
        disc_entropy = (-probs * probs.log() - (1 - probs) * (1 - probs).log()).mean().item()

        def grad_norm(module: nn.Module) -> float:
            norms = [p.grad.norm(2).item() for p in module.parameters() if p.grad is not None]
            return (sum(norms) / len(norms)) if norms else 0.0

        grad_norm_G = grad_norm(self.gen_enc) + grad_norm(self.gen_dec)
        grad_norm_D = grad_norm(self.disc_feat) + grad_norm(self.disc_out)

        metrics: Dict[str, float] = {
            "gan/D_conf_real": D_conf_real,
            "gan/D_conf_fake": D_conf_fake,
            "gan/D_conf_fake_Gview": D_conf_fake_Gview,
            "gan/cosine_mean_similarity": float(cosine_sim),
            "gan/std_diff": float(std_diff),
            "gan/adv_balance": float(adv_balance),
            "gan/disc_entropy": float(disc_entropy),
            "gan/grad_norm_G": float(grad_norm_G),
            "gan/grad_norm_D": float(grad_norm_D),
        }
        wandb.log(metrics, step=step)
        return metrics

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(
        self,
        z_in: torch.Tensor,
        z_target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Parameters
        ----------
        z_in : torch.Tensor [B, dim_in]
        z_target : Optional[torch.Tensor] [B, dim_out]
        kwargs :
            - current_step: int (for wandb)
            - log_to_wandb: bool

        Returns
        -------
        Dict[str, Any]
            {
              "tilde": torch.Tensor [B, dim_out],
              "losses": Dict[str, torch.Tensor]
            }
        """
        z_hat = self.generate(z_in)
        if z_target is None:
            return {"tilde": z_hat, "losses": {}}

        losses, z_fake = self.compute_losses(z_in, z_target, current_step=kwargs.get("current_step", None))

        if "current_step" in kwargs and kwargs.get("log_to_wandb", True):
            step = int(kwargs["current_step"]) + 1
            self.wandb_log(
                z_in=z_in,
                z_target=z_target,
                z_fake=z_fake,
                real_pred=self.discriminate(z_in, z_target),
                fake_pred=self.discriminate(z_in, z_fake.detach()),
                fake_pred_g=self.discriminate(z_in, z_fake),
                step=step,
            )

        return {"tilde": z_hat, "losses": losses}



# ------------------------------------------------------------
# Utilities: noise schedule
# ------------------------------------------------------------
def make_beta_schedule(
        num_steps: int,
        schedule_type: str = "cosine",
        max_beta: float = 0.999
) -> torch.Tensor:
    """
    Returns a beta schedule of shape [num_steps]
    """
    if schedule_type == "linear":
        betas = torch.linspace(1e-4, max_beta, num_steps)
        return betas

    if schedule_type == "cosine":
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos((x / num_steps + 0.008) / 1.008 * (3.141592653589793 / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # ⭐ critical fix: do NOT allow ᾱₜ to go too close to zero
        min_signal_rate = 0.1
        alphas_cumprod = alphas_cumprod.clamp_min(min_signal_rate)

        # recompute betas from clamped ᾱₜ
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(1e-8, 1e-3)

        return betas

    raise ValueError(f"Unknown schedule_type '{schedule_type}'")


# ------------------------------------------------------------
# The Model
# ------------------------------------------------------------
class Z1Generator_Diffusion(nn.Module):
    """
    Conditional DDPM for latent → latent mapping:
        Z_in  (e.g. Z2)
        =>
        Z_target (e.g. Z1)

    We model q(z_t | z_0) with a standard forward noising process
    and learn ε_θ(z_t, z_in, t). Sampling defines an implicit
    p_θ(z_target | z_in) via the reverse diffusion chain.
    """

    def __init__(self, args: Dict[str, Any], **kwargs) -> None:
        super().__init__()

        # ----------------------------------------------------
        # Basic config
        # ----------------------------------------------------
        self.args = args
        self.dim_in = args.get("dim_in", 128)   # conditioning vector (Z2)
        self.dim_out = args.get("dim_out", 128) # target vector (Z1)
        self.hidden_dim = args.get("hidden_dim", 256)
        self.time_embed_dim = args.get("time_embed_dim", 128)
        self.num_steps = args.get("num_steps", 1000)
        self.beta_schedule = args.get("beta_schedule", "cosine")

        # noise schedule
        betas = make_beta_schedule(self.num_steps, self.beta_schedule)
        self.register_buffer("betas", betas)                          # [T]
        self.register_buffer("alphas", 1.0 - betas)                   # [T]
        self.register_buffer("alphas_cumprod",
                             torch.cumprod(1.0 - betas, dim=0))      # [T]
        self.register_buffer("alphas_cumprod_prev",
                             F.pad(self.alphas_cumprod[:-1],
                                   (1, 0), value=1.0))               # [T]
        self.register_buffer("sqrt_alphas_cumprod",
                             torch.sqrt(self.alphas_cumprod))        # [T]
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - self.alphas_cumprod))  # [T]

        # ----------------------------------------------------
        # Time embedding (Sinusoidal)
        # ----------------------------------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        # ----------------------------------------------------
        # The neural network εθ(z_noisy, z_in, t)
        # ----------------------------------------------------
        input_dim = self.dim_out + self.dim_in + self.hidden_dim  # noisy z + conditioning + time embed
        self.eps_model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.dim_out),
        )

    # ------------------------------------------------------------
    # Time embedding (sinusoidal)
    # ------------------------------------------------------------
    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Standard sinusoidal time embedding.
        timesteps: [B] int64
        returns: [B, time_embed_dim]
        """
        device = timesteps.device
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -torch.arange(0, half, device=device).float()
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = timesteps[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

    # ------------------------------------------------------------
    # Forward diffusion: sample z_t from q(z_t | z_0)
    # ------------------------------------------------------------
    def q_sample(
        self,
        z_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample z_t from q(z_t | z_0).
        """
        if noise is None:
            noise = torch.randn_like(z_start)

        sqrt_ac = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_ac * z_start + sqrt_omc * noise

    # ------------------------------------------------------------
    # DDPM training loss: predict noise ε
    # ------------------------------------------------------------
    def compute_losses(
        self,
        z_in: torch.Tensor,
        z_target: torch.Tensor,
        current_step: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        B = z_target.size(0)
        device = z_target.device

        # random time-step
        t = torch.randint(0, self.num_steps, (B,), device=device).long()
        noise = torch.randn_like(z_target)

        # forward noising
        z_noisy = self.q_sample(z_target, t, noise)

        # time embedding
        t_emb = self.time_mlp(self.timestep_embedding(t))

        # concatenate everything
        model_in = torch.cat([z_noisy, z_in, t_emb], dim=-1)

        # model predicts noise
        noise_pred = self.eps_model(model_in)

        # MSE loss between true noise and predicted noise
        loss = F.mse_loss(noise_pred, noise)

        losses = {
            "total": loss,
            "noise_pred_loss": loss.detach(),
        }

        # return losses and the noisy sample (optional)
        return losses, z_noisy

    # ------------------------------------------------------------
    # Posterior p(z_{t-1} | z_t, z_in) ≈ q(z_{t-1} | z_t, z0_pred)
    # (DDPM-style; very similar to what Stable Diffusion schedulers do)
    # ------------------------------------------------------------
    def p_mean_variance(
        self,
        z_t: torch.Tensor,
        z_in: torch.Tensor,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute:
          - ε_θ(z_t, z_in, t)
          - z0_pred
          - posterior mean / variance of q(z_{t-1} | z_t, z0_pred)

        z_t: [B, D]
        z_in: [B, D_in]
        t: [B] long
        """
        # time embedding
        t_emb = self.time_mlp(self.timestep_embedding(t))

        # predict noise
        model_in = torch.cat([z_t, z_in, t_emb], dim=-1)
        eps = self.eps_model(model_in)

        # gather schedule values
        betas_t = self.betas[t].unsqueeze(-1)                     # [B, 1]
        alphas_t = self.alphas[t].unsqueeze(-1)                   # [B, 1]
        alpha_cum_t = self.alphas_cumprod[t].unsqueeze(-1)        # [B, 1]
        alpha_cum_prev = self.alphas_cumprod_prev[t].unsqueeze(-1)# [B, 1]

        alpha_cum_t = torch.clamp(alpha_cum_t, 1e-5, 1.0)
        sqrt_alpha_cum_t = torch.sqrt(alpha_cum_t)
        sqrt_one_minus = torch.sqrt(torch.clamp(
            1.0 - alpha_cum_t, min=1e-5
        ))

        # DDPM-style prediction of z0
        z0_pred = (z_t - sqrt_one_minus * eps) / (sqrt_alpha_cum_t + 1e-5)
        z0_pred = torch.clamp(z0_pred, -10.0, 10.0)

        # posterior variance q(z_{t-1} | z_t, z0)
        # Eq. (4) in Ho et al. 2020
        posterior_variance = (
            betas_t * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum_t)
        )
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        posterior_std = torch.sqrt(posterior_variance)

        # posterior mean
        posterior_mean = (
            torch.sqrt(alpha_cum_prev) * betas_t / (1.0 - alpha_cum_t) * z0_pred
            + torch.sqrt(alphas_t) * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum_t) * z_t
        )

        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "std": posterior_std,
            "z0_pred": z0_pred,
            "eps": eps,
        }

    # ------------------------------------------------------------
    # Reverse sampling (DDPM-like, StableDiffusion-ish process)
    # ------------------------------------------------------------
    @torch.no_grad()
    def sample_z(
        self,
        z_in: torch.Tensor,
        num_steps: Optional[int] = None,
        eta: float = 1.0,
        return_chain: bool = False,
    ) -> torch.Tensor:
        """
        Generate a sample given conditioning z_in.

        eta=1.0  → pure DDPM (full noise in each step)
        eta=0.0  → deterministic DDIM-like behaviour (uses z0_pred; no extra noise)

        If return_chain=True, also returns all intermediate z_t's.
        """
        device = z_in.device
        T = num_steps if num_steps is not None else self.num_steps

        B = z_in.size(0)
        z_t = torch.randn(B, self.dim_out, device=device)

        chain = [z_t]

        for i in reversed(range(T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            out = self.p_mean_variance(z_t, z_in, t)
            mean = out["mean"]
            std = out["std"]
            z0_pred = out["z0_pred"]

            if i > 0:
                if eta > 0.0:
                    # stochastic DDPM-style step
                    noise = torch.randn_like(z_t)
                    z_t = mean + eta * std * noise
                else:
                    # deterministic "DDIM-like" step: go to mean of posterior
                    z_t = mean
            else:
                # final step: directly use z0_pred as z_target sample
                z_t = z0_pred

            z_t = torch.clamp(z_t, -20.0, 20.0)
            if return_chain:
                chain.append(z_t)

        if return_chain:
            return z_t, torch.stack(chain, dim=1)  # [B, T+1, D]
        return z_t

    # ------------------------------------------------------------
    # A forward() wrapper for your existing training pipeline
    # ------------------------------------------------------------
    def forward(
        self,
        z_in: torch.Tensor,
        z_target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Mirrors your GAN forward() API.
        - If z_target is provided → training mode
        - If z_target is None → sampling mode
        """
        if z_target is None:
            # inference mode
            z_hat = self.sample_z(z_in)
            return {"tilde": z_hat, "losses": {}}

        # training mode (compute DDPM noise loss)
        losses, _ = self.compute_losses(
            z_in=z_in,
            z_target=z_target,
            current_step=kwargs.get("current_step", None)
        )

        # for consistency with GAN API
        z_hat = self.sample_z(z_in)
        return {"tilde": z_hat, "losses": losses}




class CrossAttentionEpsModel(nn.Module):
    def __init__(
        self,
        dim_out: int,
        dim_in: int,
        time_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_len: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.z_t_proj  = nn.Linear(dim_out, hidden_dim)
        self.z_in_proj = nn.Linear(dim_in,  hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        self.pos_emb_out = nn.Embedding(max_len, hidden_dim)
        self.pos_emb_in  = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn_layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, dim_out)

    def forward(self, z_t: torch.Tensor, z_in: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, _, T_out = z_t.shape
        _, _, T_in  = z_in.shape

        z_t_seq  = z_t.transpose(1, 2)
        z_in_seq = z_in.transpose(1, 2)

        z_t_h  = self.z_t_proj(z_t_seq)
        z_in_h = self.z_in_proj(z_in_seq)

        pos_out = self.pos_emb_out(torch.arange(T_out, device=z_t.device))
        pos_in  = self.pos_emb_in(torch.arange(T_in,  device=z_in.device))

        z_t_h = z_t_h + pos_out.unsqueeze(0)
        z_in_h = z_in_h + pos_in.unsqueeze(0)

        t_bias = self.time_proj(t_emb).unsqueeze(1)
        z_t_h = z_t_h + t_bias

        z_t_h = self.self_attn_layers(z_t_h)

        z_t_ca, _ = self.cross_attn(
            query=z_t_h,
            key=z_in_h,
            value=z_in_h,
        )
        z_t_h = self.cross_attn_norm(z_t_h + z_t_ca)

        eps_seq = self.output_proj(z_t_h)
        return eps_seq.transpose(1, 2)
class LatentDiffusionTransformer(nn.Module):
    """
    Conditional DDPM with Transformer ε-model:
      z_in     : [B, F_in,  T_in]
      z_target : [B, F_out, T_out]

    Includes dataset-level latent normalization.
    """

    def __init__(self, args: Dict[str, Any], **kwargs) -> None:
        super().__init__()

        # ----------------------------------------------------
        # Basic configuration
        # ----------------------------------------------------
        self.args = args
        self.dim_in  = args.get("dim_in", 128)
        self.dim_out = args.get("dim_out", 128)
        self.hidden_dim = args.get("hidden_dim", 256)
        self.time_embed_dim = args.get("time_embed_dim", 128)
        self.num_steps = args.get("num_steps", 1000)
        self.beta_schedule = args.get("beta_schedule", "cosine")
        self.freeze_stats_after_epoch = args.get("freeze_stats_epoch", 30)

        # ==============================================================
        # Dataset-level latent normalization buffers
        # ==============================================================

        # User can provide dataset-level mean/std in args
        # shape: [1, F, 1]
        dataset_z_in_mean  = args.get("z_in_mean",  None)
        dataset_z_in_std   = args.get("z_in_std",   None)
        dataset_z_tgt_mean = args.get("z_target_mean", None)
        dataset_z_tgt_std  = args.get("z_target_std",  None)

        # Initialize buffers
        self.register_buffer(
            "z_in_mean",
            torch.zeros(1, self.dim_in, 1) if dataset_z_in_mean is None
            else dataset_z_in_mean.clone()
        )
        self.register_buffer(
            "z_in_std",
            torch.ones(1, self.dim_in, 1) if dataset_z_in_std is None
            else dataset_z_in_std.clone()
        )

        self.register_buffer(
            "z_target_mean",
            torch.zeros(1, self.dim_out, 1) if dataset_z_tgt_mean is None
            else dataset_z_tgt_mean.clone()
        )
        self.register_buffer(
            "z_target_std",
            torch.ones(1, self.dim_out, 1) if dataset_z_tgt_std is None
            else dataset_z_tgt_std.clone()
        )

        # Whether we should update stats via EMA
        self.use_ema_stats = (
            dataset_z_in_mean is None or dataset_z_tgt_mean is None
        )

        self.latent_momentum = args.get("latent_momentum", 0.01)

        # ==============================================================
        # Beta schedule
        # ==============================================================
        betas = make_beta_schedule(self.num_steps, self.beta_schedule)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev",
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0),
        )
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod)
        )

        # ==============================================================
        # Time embedding
        # ==============================================================
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        # ==============================================================
        # Transformer ε-model
        # ==============================================================
        self.eps_model = CrossAttentionEpsModel(
            dim_out=self.dim_out,
            dim_in=self.dim_in,
            time_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=args.get("transformer_layers", 4),
            num_heads=args.get("transformer_heads", 8),
        )

    # ==================================================================
    # Normalization utilities
    # ==================================================================
    @torch.no_grad()
    def update_latent_stats_ema(self, z_target, z_in, **kwargs):
        """EMA update for dataset statistics (only if no dataset stats provided)."""
        if not self.training:
            return
        if not self.use_ema_stats:
            return
        if kwargs["current_step"] >= self.freeze_stats_after_epoch:
            return

        m = self.latent_momentum

        # ---- target latent stats ----
        tgt_mean = z_target.mean(dim=(0, 2), keepdim=True)
        tgt_std  = z_target.std(dim=(0, 2), keepdim=True) + 1e-6

        self.z_target_mean.copy_( (1 - m) * self.z_target_mean + m * tgt_mean )
        self.z_target_std.copy_(  (1 - m) * self.z_target_std  + m * tgt_std )

        # ---- conditioning latent stats ----
        in_mean = z_in.mean(dim=(0, 2), keepdim=True)
        in_std  = z_in.std(dim=(0, 2), keepdim=True) + 1e-6

        self.z_in_mean.copy_( (1 - m) * self.z_in_mean + m * in_mean )
        self.z_in_std.copy_(  (1 - m) * self.z_in_std  + m * in_std )

    def normalize(self, z, mean, std):
        return (z - mean) / (std + 1e-6)

    def denormalize(self, z, mean, std):
        return z * std + mean

    # ==================================================================
    # Sinusoidal time embedding
    # ==================================================================
    def timestep_embedding(self, t):
        device = t.device
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -torch.arange(0, half, device=device)
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # ==================================================================
    # Forward diffusion q(z_t | z_0)
    # ==================================================================
    def q_sample(self, z_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z_start)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ac * z_start + sqrt_omc * noise

    # ==================================================================
    # Training loss
    # ==================================================================
    def compute_losses(self, z_in, z_target, **kwargs):


        # --- update stats if needed ---
        self.update_latent_stats_ema(z_target, z_in, **kwargs)

        # --- normalize latents ---
        z_in_n     = self.normalize(z_in,     self.z_in_mean,     self.z_in_std)
        z_target_n = self.normalize(z_target, self.z_target_mean, self.z_target_std)

        B = z_target.size(0)
        device = z_target.device
        t = torch.randint(0, self.num_steps, (B,), device=device).long()

        noise_scale = 0.1  # try 0.1 first
        noise = noise_scale * torch.randn_like(z_target_n)
        # noise = torch.randn_like(z_target_n)
        z_noisy = self.q_sample(z_target_n, t, noise)

        t_emb = self.time_mlp(self.timestep_embedding(t))
        noise_pred = self.eps_model(z_noisy, z_in_n, t_emb)

        loss = F.mse_loss(noise_pred, noise)

        return {"total": loss, "noise_pred_loss": loss.detach()}

    # ==================================================================
    # Debug: overfit a single example with fixed t and noise
    # ==================================================================
    def compute_losses_debug(self, z_in, z_target):
        """
        Overfit mode:
        - fix t = 0
        - fix one noise sample
        - no randomization across iterations
        So the model just learns a deterministic mapping.
        """

        # use dataset/EMA stats
        z_in_n     = self.normalize(z_in,     self.z_in_mean,     self.z_in_std)
        z_target_n = self.normalize(z_target, self.z_target_mean, self.z_target_std)

        B = z_target_n.size(0)
        device = z_target_n.device

        # fix t = 0 for all
        t = torch.zeros(B, dtype=torch.long, device=device)

        # --------------------------------------------------
        # Create and CACHE a single noise + z_noisy pair
        # --------------------------------------------------
        if not hasattr(self, "_debug_cache"):
            noise = torch.randn_like(z_target_n)
            z_noisy = self.q_sample(z_target_n, t, noise)
            # cache detached copies so they don't change over time
            self._debug_cache = {
                "z_in_n":   z_in_n.detach(),
                "z_noisy":  z_noisy.detach(),
                "noise":    noise.detach(),
                "t":        t.detach(),
            }

        cache = self._debug_cache
        z_in_n_cached = cache["z_in_n"]
        z_noisy_cached = cache["z_noisy"]
        noise_cached = cache["noise"]
        t_cached = cache["t"]

        # time embedding of fixed t
        t_emb = self.time_mlp(self.timestep_embedding(t_cached))

        # eps prediction
        noise_pred = self.eps_model(z_noisy_cached, z_in_n_cached, t_emb)

        loss = F.mse_loss(noise_pred, noise_cached)
        return {"total": loss, "noise_pred_loss": loss.detach()}

    def compute_losses_debug_random_noise(self, z_in, z_target):
        z_in_n = self.normalize(z_in, self.z_in_mean, self.z_in_std)
        z_target_n = self.normalize(z_target, self.z_target_mean, self.z_target_std)

        B = z_target_n.size(0)
        device = z_target_n.device

        # fix timestep
        t = torch.zeros(B, dtype=torch.long, device=device)

        # fresh noise every iteration
        noise_scale = 1.2  # try 0.1 first
        noise = noise_scale * torch.randn_like(z_target_n)
        z_noisy = self.q_sample(z_target_n, t, noise)

        t_emb = self.time_mlp(self.timestep_embedding(t))
        noise_pred = self.eps_model(z_noisy, z_in_n, t_emb)

        loss = F.mse_loss(noise_pred, noise)
        return {"total": loss, "noise_pred_loss": loss.detach()}

    def compute_losses_debug_random_t(self, z_in, z_target):
        z_in_n = self.normalize(z_in, self.z_in_mean, self.z_in_std)
        z_target_n = self.normalize(z_target, self.z_target_mean, self.z_target_std)

        B = z_target_n.size(0)
        device = z_target_n.device

        # sample random timesteps
        t = torch.randint(0, self.num_steps, (B,), device=device)

        # cache only noise (noisy varies with t)
        if not hasattr(self, "_debug_noise"):
            self._debug_noise = torch.randn_like(z_target_n)

        noise = self._debug_noise
        z_noisy = self.q_sample(z_target_n, t, noise)

        t_emb = self.time_mlp(self.timestep_embedding(t))
        noise_pred = self.eps_model(z_noisy, z_in_n, t_emb)

        loss = F.mse_loss(noise_pred, noise)
        return {"total": loss, "noise_pred_loss": loss.detach()}

    # ==================================================================
    # Reverse diffusion posterior
    # ==================================================================
    def p_mean_variance(self, z_t, z_in, t):

        t_emb = self.time_mlp(self.timestep_embedding(t))
        eps = self.eps_model(z_t, z_in, t_emb)

        betas_t = self.betas[t].view(-1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1)
        alpha_cum_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_cum_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)

        alpha_cum_t = torch.clamp(alpha_cum_t, 1e-5, 1.0)
        sqrt_alpha_cum_t = torch.sqrt(alpha_cum_t)
        sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_cum_t, 1e-5))

        z0_pred = (z_t - sqrt_one_minus * eps) / (sqrt_alpha_cum_t + 1e-5)
        z0_pred = torch.clamp(z0_pred, -10.0, 10.0)

        posterior_variance = betas_t * (1 - alpha_cum_prev) / (1 - alpha_cum_t)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        posterior_std = torch.sqrt(posterior_variance)

        posterior_mean = (
            torch.sqrt(alpha_cum_prev) * betas_t / (1 - alpha_cum_t) * z0_pred +
            torch.sqrt(alphas_t) * (1 - alpha_cum_prev) / (1 - alpha_cum_t) * z_t
        )

        return {
            "mean": posterior_mean,
            "std": posterior_std,
            "z0_pred": z0_pred
        }

    # ==================================================================
    # Sampling
    # ==================================================================
    @torch.no_grad()
    def sample_z(self, z_in, num_steps=None, eta=1.0, return_chain=False):

        # --- normalize conditioning ---
        z_in_n = self.normalize(z_in, self.z_in_mean, self.z_in_std)

        T_steps = num_steps if num_steps is not None else self.num_steps
        device = z_in.device

        B, _, T_in = z_in_n.shape
        T_out = T_in     # adapt externally if needed
        z_t = torch.randn(B, self.dim_out, T_out, device=device)

        chain = [z_t] if return_chain else None

        for i in reversed(range(T_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            out = self.p_mean_variance(z_t, z_in_n, t)
            mean, std, z0_pred = out["mean"], out["std"], out["z0_pred"]

            if i > 0:
                if eta > 0:
                    z_t = mean + eta * std * torch.randn_like(z_t)
                else:
                    z_t = mean
            else:
                z_t = z0_pred

            z_t = torch.clamp(z_t, -20, 20)
            if return_chain:
                chain.append(z_t)

        # --- denormalize output ---
        z_t = self.denormalize(z_t, self.z_target_mean, self.z_target_std)

        if return_chain:
            return z_t, torch.stack(chain, dim=1)

        return z_t

    # ==================================================================
    # Forward wrapper
    # ==================================================================
    def forward(self, z_in, z_target=None, **kwargs):

        if z_target is None:
            return {"tilde": self.sample_z(z_in), "losses": {}}

        # losses = self.compute_losses_debug(z_in, z_target)
        # losses = self.compute_losses_debug_random_noise(z_in, z_target)
        # losses = self.compute_losses_debug_random_t(z_in, z_target)
        losses = self.compute_losses(z_in, z_target, **kwargs)
        z_hat = self.sample_z(z_in)
        return {"tilde": z_hat, "losses": losses}

def make_beta_schedule_mlp(num_steps: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, num_steps)
    if schedule == "cosine":
        steps = num_steps + 1
        t = torch.linspace(0, num_steps, steps) / num_steps
        s = 0.008
        alphas_cumprod = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"Unknown schedule: {schedule}")


def make_beta_schedule_scaled_fixed(
    num_steps: int,
    schedule: str = "cosine",
    alpha_bar_target: float = 1e-3,
) -> torch.Tensor:
    """
    Take your original beta schedule (cosine/linear) and scale it so that
    the final alpha_bar ≈ alpha_bar_target.

    This preserves the *shape* of the schedule but makes the overall noise
    less or more aggressive depending on target.
    """
    betas_raw = make_beta_schedule_mlp(num_steps, schedule=schedule)
    betas_raw = betas_raw.clamp(1e-8, 0.999)

    def compute_alpha_bar(k: float) -> float:
        alphas = 1 - k * betas_raw
        alphas = alphas.clamp(1e-8, 0.999)
        ac = torch.cumprod(alphas, dim=0)[-1]
        return float(ac.item())

    # binary search on scale k
    low, high = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if compute_alpha_bar(mid) > alpha_bar_target:
            # too little noise -> increase k
            low = mid
        else:
            # too much noise -> decrease k
            high = mid
    k = 0.5 * (low + high)

    betas = (k * betas_raw).clamp(1e-8, 0.999)
    return betas



class MLPEpsModel(nn.Module):
    def __init__(
        self,
        dim_out: int,
        dim_in: int,
        time_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        input_dim = dim_out + dim_in + time_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, dim_out))

        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_in: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, z_in, t_emb], dim=-1)
        return self.net(x)
class LatentDiffusionMLP(nn.Module):
    def __init__(self, args: Dict[str, Any], **kwargs) -> None:
        super().__init__()
        self.args = args
        self.dim_in = args.get("dim_in", 128)
        self.dim_out = args.get("dim_out", 128)
        self.hidden_dim = args.get("hidden_dim", 256)
        self.time_embed_dim = args.get("time_embed_dim", 128)
        self.num_steps = args.get("num_steps", 1000)
        self.beta_schedule = args.get("beta_schedule", "cosine")
        self.freeze_stats_after_epoch = args.get("freeze_stats_epoch", 30)
        self.latent_momentum = args.get("latent_momentum", 0.01)
        self.noise_scale = args.get("noise_scale", 1.0)

        dataset_z_in_mean = args.get("z_in_mean", None)
        dataset_z_in_std = args.get("z_in_std", None)
        dataset_z_tgt_mean = args.get("z_target_mean", None)
        dataset_z_tgt_std = args.get("z_target_std", None)

        self.register_buffer(
            "z_in_mean",
            torch.zeros(1, self.dim_in) if dataset_z_in_mean is None else dataset_z_in_mean.clone()
        )
        self.register_buffer(
            "z_in_std",
            torch.ones(1, self.dim_in) if dataset_z_in_std is None else dataset_z_in_std.clone()
        )
        self.register_buffer(
            "z_target_mean",
            torch.zeros(1, self.dim_out) if dataset_z_tgt_mean is None else dataset_z_tgt_mean.clone()
        )
        self.register_buffer(
            "z_target_std",
            torch.ones(1, self.dim_out) if dataset_z_tgt_std is None else dataset_z_tgt_std.clone()
        )

        self.use_ema_stats = dataset_z_in_mean is None or dataset_z_tgt_mean is None

        betas = make_beta_schedule_scaled_fixed(self.num_steps, self.beta_schedule)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev",
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0),
        )
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.eps_model = MLPEpsModel(
            dim_out=self.dim_out,
            dim_in=self.dim_in,
            time_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=args.get("mlp_layers", 4),
        )

    @torch.no_grad()
    def update_latent_stats_ema(self, z_target: torch.Tensor, z_in: torch.Tensor, current_step: int = 0):
        if not self.training:
            return
        if not self.use_ema_stats:
            return
        if current_step >= self.freeze_stats_after_epoch:
            return

        m = self.latent_momentum

        tgt_mean = z_target.mean(dim=0, keepdim=True)
        tgt_std = z_target.std(dim=0, keepdim=True) + 1e-6
        self.z_target_mean.copy_((1 - m) * self.z_target_mean + m * tgt_mean)
        self.z_target_std.copy_((1 - m) * self.z_target_std + m * tgt_std)

        in_mean = z_in.mean(dim=0, keepdim=True)
        in_std = z_in.std(dim=0, keepdim=True) + 1e-6
        self.z_in_mean.copy_((1 - m) * self.z_in_mean + m * in_mean)
        self.z_in_std.copy_((1 - m) * self.z_in_std + m * in_std)

    def normalize(self, z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (z - mean) / (std + 1e-6)

    def denormalize(self, z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return z * std + mean

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            -torch.arange(0, half, device=device)
            * (torch.log(torch.tensor(10000.0, device=device)) / (half - 1))
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def q_sample(self, z_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z_start)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_ac * z_start + sqrt_omc * noise

    def compute_losses(self, z_in: torch.Tensor, z_target: torch.Tensor, **kwargs):
        current_step = kwargs.get("current_step", 0)
        self.update_latent_stats_ema(z_target, z_in, current_step)

        z_in_n = self.normalize(z_in, self.z_in_mean, self.z_in_std)
        z_target_n = self.normalize(z_target, self.z_target_mean, self.z_target_std)

        B = z_target.size(0)
        device = z_target.device
        t = torch.randint(0, self.num_steps, (B,), device=device).long()

        noise = self.noise_scale * torch.randn_like(z_target_n)
        z_noisy = self.q_sample(z_target_n, t, noise)

        t_emb = self.time_mlp(self.timestep_embedding(t))
        noise_pred = self.eps_model(z_noisy, z_in_n, t_emb)

        loss = F.mse_loss(noise_pred, noise)
        return {"total": loss, "noise_pred_loss": loss.detach()}

    def compute_losses_debug(self, z_in: torch.Tensor, z_target: torch.Tensor):
        z_in_n = self.normalize(z_in, self.z_in_mean, self.z_in_std)
        z_target_n = self.normalize(z_target, self.z_target_mean, self.z_target_std)

        B = z_target_n.size(0)
        device = z_target_n.device
        t = torch.zeros(B, dtype=torch.long, device=device)

        if not hasattr(self, "_debug_cache"):
            noise = torch.randn_like(z_target_n)
            z_noisy = self.q_sample(z_target_n, t, noise)
            self._debug_cache = {
                "z_in_n": z_in_n.detach(),
                "z_noisy": z_noisy.detach(),
                "noise": noise.detach(),
                "t": t.detach(),
            }

        cache = self._debug_cache
        z_in_n_cached = cache["z_in_n"]
        z_noisy_cached = cache["z_noisy"]
        noise_cached = cache["noise"]
        t_cached = cache["t"]

        t_emb = self.time_mlp(self.timestep_embedding(t_cached))
        noise_pred = self.eps_model(z_noisy_cached, z_in_n_cached, t_emb)

        loss = F.mse_loss(noise_pred, noise_cached)
        return {"total": loss, "noise_pred_loss": loss.detach()}

    def compute_losses_debug_random_t(self, z_in: torch.Tensor, z_target: torch.Tensor):
        z_in_n = self.normalize(z_in, self.z_in_mean, self.z_in_std)
        z_target_n = self.normalize(z_target, self.z_target_mean, self.z_target_std)

        B = z_target_n.size(0)
        device = z_target_n.device
        t = torch.randint(0, self.num_steps, (B,), device=device)

        if not hasattr(self, "_debug_noise"):
            self._debug_noise = torch.randn_like(z_target_n)

        noise = self._debug_noise
        z_noisy = self.q_sample(z_target_n, t, noise)

        t_emb = self.time_mlp(self.timestep_embedding(t))
        noise_pred = self.eps_model(z_noisy, z_in_n, t_emb)

        loss = F.mse_loss(noise_pred, noise)
        return {"total": loss, "noise_pred_loss": loss.detach()}

    def p_mean_variance(self, z_t: torch.Tensor, z_in: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_mlp(self.timestep_embedding(t))
        eps = self.eps_model(z_t, z_in, t_emb)

        betas_t = self.betas[t].view(-1, 1)
        alphas_t = self.alphas[t].view(-1, 1)
        alpha_cum_t = self.alphas_cumprod[t].view(-1, 1)
        alpha_cum_prev = self.alphas_cumprod_prev[t].view(-1, 1)

        alpha_cum_t = torch.clamp(alpha_cum_t, 1e-5, 1.0)
        sqrt_alpha_cum_t = torch.sqrt(alpha_cum_t)
        sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_cum_t, 1e-5))

        z0_pred = (z_t - sqrt_one_minus * eps) / (sqrt_alpha_cum_t + 1e-5)
        z0_pred = torch.clamp(z0_pred, -10.0, 10.0)

        posterior_variance = betas_t * (1 - alpha_cum_prev) / (1 - alpha_cum_t)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        posterior_std = torch.sqrt(posterior_variance)

        posterior_mean = (
            torch.sqrt(alpha_cum_prev) * betas_t / (1 - alpha_cum_t) * z0_pred +
            torch.sqrt(alphas_t) * (1 - alpha_cum_prev) / (1 - alpha_cum_t) * z_t
        )

        return {
            "mean": posterior_mean,
            "std": posterior_std,
            "z0_pred": z0_pred,
        }

    @torch.no_grad()
    def sample_z(self, z_in: torch.Tensor, num_steps: int | None = None, eta: float = 1.0, return_chain: bool = False):
        z_in_n = self.normalize(z_in, self.z_in_mean, self.z_in_std)

        T_steps = num_steps if num_steps is not None else self.num_steps
        device = z_in.device

        B, _ = z_in_n.shape
        z_t = torch.randn(B, self.dim_out, device=device)

        chain = [z_t] if return_chain else None

        for i in reversed(range(T_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            out = self.p_mean_variance(z_t, z_in_n, t)
            mean, std, z0_pred = out["mean"], out["std"], out["z0_pred"]

            if i > 0:
                if eta > 0:
                    z_t = mean + eta * std * torch.randn_like(z_t)
                else:
                    z_t = mean
            else:
                z_t = z0_pred

            z_t = torch.clamp(z_t, -20, 20)
            if return_chain:
                chain.append(z_t)

        z_t = self.denormalize(z_t, self.z_target_mean, self.z_target_std)

        if return_chain:
            return z_t, torch.stack(chain, dim=1)

        return z_t

    def forward(self, z_in: torch.Tensor, z_target: torch.Tensor | None = None, **kwargs):
        if z_target is None:
            return {"tilde": self.sample_z(z_in), "losses": {}}

        losses = self.compute_losses(z_in, z_target, **kwargs)
        z_hat = self.sample_z(z_in)
        return {"tilde": z_hat, "losses": losses}
class SimpleZ2toZ1(nn.Module):
    def __init__(self, dim_in, dim_out, hidden=512, depth=3):
        super().__init__()
        layers = []
        dims = [dim_in] + [hidden] * depth + [dim_out]
        for d1, d2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))
            if d2 != dim_out:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z2, z1_target=None, compute_loss=True, **kwargs):
        """
        Keeps the same interface as your diffusion model:
            returns {"tilde": output, "losses": {...}}
        """
        pred = self.net(z2)

        loss = F.mse_loss(pred, z1_target)
        return {
            "tilde" :pred,
            "losses": {"total": loss.detach(), "noise_pred_loss": loss.detach()}
        }
class Model_GeneratorMLP(nn.Module):
    def __init__(self, args, encs):
        super(Model_GeneratorMLP, self).__init__()

        self.args = args
        self.norm_decision = args.get("norm_decision", False)

        d_model = args.d_model
        num_classes = args.num_classes
        dropout = args.dropout
        fc_inner = args.fc_inner
        self.num_classes = args.num_classes
        self.cls_type = args.get("cls_type", "tf")
        self.perturb = {"type": args.perturb}

        # the original encoders (kept as-is)
        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        # ---------------------------------------------------
        #  REPLACE diffusion enc_2 / enc_3 WITH SIMPLE MLPs
        # ---------------------------------------------------
        dim_z = args.get("latent_dim", 512)  # ensure this matches your z-dim

        self.enc_2 = SimpleZ2toZ1(dim_in=dim_z, dim_out=dim_z)
        self.enc_3 = SimpleZ2toZ1(dim_in=dim_z, dim_out=dim_z)

        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )
            self.mu_head = nn.Linear(d_model, num_classes)
            self.logvar_head = nn.Linear(d_model, num_classes)

        elif self.cls_type == "conformer":

            self.common_fc_1 =TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6, output_dim=d_model)

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )

        else:
            raise ValueError("Unknown cls_type")

        self.mu_head = nn.Linear(fc_inner, num_classes)
        self.logvar_head = nn.Linear(fc_inner, num_classes)



    # -------------------------------------------------------
    # feature extraction
    # -------------------------------------------------------

    def _get_features(self, x, **kwargs):
        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)
        return a, v


    # -------------------------------------------------------
    # forward pass without perturbation
    # -------------------------------------------------------

    def _forward_main(self, a, v, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"],
                                                    v["features"]["combined"]), dim=1))
            feat = self.common_fc_2(aggr_feat)

        elif self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1(
                [a["nonaggr_features"]["combined"],
                 v["nonaggr_features"]["combined"]],
                return_all=True
            )
            feat = self.common_fc_2(aggr_feat)

        pred_mu = self.mu_head(feat)
        pred_logvar = self.logvar_head(feat)

        if kwargs.get("stochastic", False):
            std = torch.exp(0.5 * pred_logvar)
            eps = torch.randn_like(std)
            pred = pred_mu + eps * std
            return pred, pred_mu, pred_logvar

        return pred_mu, aggr_feat


    # -------------------------------------------------------
    # perturbation function (MLP replaces diffusion)
    # -------------------------------------------------------

    def _perturb_features(self, z1, z2, **kwargs):

        if self.perturb["type"] == "zeros":
            return {"tilde": torch.zeros_like(z1).detach()}

        elif self.perturb["type"] == "cmn":
            mask = (torch.rand_like(z1) < self.perturb.get("p", 0.5)).float()
            scale = self.perturb.get("s", 0.1) * (
                1 + torch.tanh(z2.norm(dim=-1, keepdim=True) /
                               self.perturb.get("c", 1))
            )
            noise = torch.randn_like(z1) * scale
            tilde = mask * z1 + (1 - mask) * noise
            return {"tilde": tilde.detach()}

        # ---------------------------------------------------
        #  HERE: enc_2 and enc_3 are now SIMPLE MLP BASELINES
        # ---------------------------------------------------
        elif self.perturb["type"] in ["gen", "gen_v2", "gennoise"]:

            if kwargs.get("swap_direction", False):
                out = self.enc_2(z2, z1_target=z1,
                                 compute_loss=self.training, **kwargs)
                wandb_loss = {"px1": out["losses"]}

            else:
                out = self.enc_3(z1, z1_target=z2,
                                 compute_loss=self.training, **kwargs)
                wandb_loss = {"px2": out["losses"]}

            if "current_step" in kwargs:
                import wandb
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)

            return out

        # fallback
        return {"tilde": z1}


    # -------------------------------------------------------
    # one-time perturbation
    # -------------------------------------------------------

    def forward_once_perturbed(self, a, v, **kwargs):

        if self.cls_type == "tf":
            z1, z2 = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
        else:
            z1, z2 = a["features"]["combined"], v["features"]["combined"]

        if kwargs.get("perturb_x1", False):
            out = self._perturb_features(z1=z1, z2=z2, swap_direction=False, **kwargs)
            pert_z1 = out["tilde"].detach()
            pert_losses = {"px1": out["losses"]}

        if kwargs.get("perturb_x2", False):
            out = self._perturb_features(z1=z1, z2=z2, swap_direction=True,  **kwargs)
            pert_z2 = out["tilde"].detach()
            pert_losses = {"px2": out["losses"]}

        return {
            "features": {
                "z1": z1,
                "z2": z2,
                "pert_z1": pert_z1 if "pert_z1" in locals() else None,
                "pert_z2": pert_z2 if "pert_z2" in locals() else None
            },
            "losses": pert_losses
        }


    # -------------------------------------------------------
    # main forward
    # -------------------------------------------------------

    def forward(self, x, **kwargs):

        output = {"preds": {}, "features": {}, "losses": {}}

        a, v = self._get_features(x, **kwargs)

        output["features"]["z1"] = a["features"]["combined"]
        output["features"]["z2"] = v["features"]["combined"]

        if self.training:
            # perturb X1
            p1 = self.forward_once_perturbed(a, v,
                                             stochastic=True,
                                             perturb_x1=True,
                                             perturb_x2=False,
                                             **kwargs)
            output["losses"].update(p1["losses"])
            output["features"]["~z1"] = p1["features"]["pert_z1"]

            # perturb X2
            p2 = self.forward_once_perturbed(a, v,
                                             stochastic=True,
                                             perturb_x1=False,
                                             perturb_x2=True,
                                             **kwargs)
            output["losses"].update(p2["losses"])
            output["features"]["~z2"] = p2["features"]["pert_z2"]

        else:
            # eval mode
            p1 = self.forward_once_perturbed(a, v,
                                             stochastic=False,
                                             perturb_x1=True,
                                             perturb_x2=False,
                                             **kwargs)
            p2 = self.forward_once_perturbed(a, v,
                                             stochastic=False,
                                             perturb_x1=False,
                                             perturb_x2=True,
                                             **kwargs)

            output["features"]["~z1"] = p1["features"]["pert_z1"]
            output["features"]["~z2"] = p2["features"]["pert_z2"]

        return output
class Model_GeneratorVAE(nn.Module):
    """
    Drop-in replacement for your diffusion-based Model_Generator.
    Uses SimpleVAEZ2toZ1 for enc_2 (z2->z1) and enc_3 (z1->z2).

    Everything else remains unchanged.
    """

    def __init__(self, args, encs):
        super(Model_GeneratorVAE, self).__init__()

        self.args = args
        self.norm_decision = args.get("norm_decision", False)

        d_model = args.d_model
        dropout = args.dropout
        fc_inner = args.fc_inner
        self.num_classes = args.num_classes
        self.cls_type = args.get("cls_type", "tf")
        self.perturb = {"type": args.perturb}

        # -----------------------------------------------------
        # enc_0 and enc_1 are your original unimodal encoders
        # -----------------------------------------------------
        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        # -----------------------------------------------------
        # Replace diffusion enc_2, enc_3 with *non-collapsing* VAEs
        # -----------------------------------------------------
        dim_z = args.get("latent_dim", 512)

        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )

        elif self.cls_type == "conformer":
            self.common_fc_1 = encs[2]
            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )
        else:
            raise ValueError("Unknown cls_type")

        self.mu_head = nn.Linear(fc_inner, self.num_classes)
        self.logvar_head = nn.Linear(fc_inner, self.num_classes)

    # ========================================================
    # Feature extraction (unchanged)
    # ========================================================

    def _get_features(self, x, **kwargs):
        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)
        return a, v

    # ========================================================
    # Classification branch (unchanged)
    # ========================================================

    def _forward_main(self, a, v, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(
                torch.cat((a["features"]["combined"],
                           v["features"]["combined"]), dim=1)
            )
            feat = self.common_fc_2(aggr_feat)

        elif self.cls_type == "conformer":
            _, aggr_feat, feat = self.common_fc_1(
                [a["nonaggr_features"]["combined"],
                 v["nonaggr_features"]["combined"]],
                return_all=True
            )
            feat = self.common_fc_2(aggr_feat)

        mu = self.mu_head(feat)
        logvar = self.logvar_head(feat)

        if kwargs.get("stochastic", False):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std, mu, logvar

        return mu, aggr_feat

    # ========================================================
    # Perturbation logic — VAE-based
    # ========================================================

    def _perturb_features(self, z1, z2, **kwargs):

        if self.perturb["type"] == "zeros":
            return {"tilde": torch.zeros_like(z1).detach()}

        elif self.perturb["type"] == "cmn":
            mask = (torch.rand_like(z1) < self.perturb.get("p", 0.5)).float()
            scale = self.perturb.get("s", 0.1) * (
                    1 + torch.tanh(z2.norm(dim=-1, keepdim=True) /
                                   self.perturb.get("c", 1))
            )
            noise = torch.randn_like(z1) * scale
            tilde = mask * z1 + (1 - mask) * noise
            return {"tilde": tilde.detach()}

        # ------------------------------
        # VAE z2→z1 or z1→z2 perturbation
        # ------------------------------
        elif self.perturb["type"] in ["gen", "gen_v2", "gennoise"]:

            if kwargs.get("swap_direction", False):
                out = self.enc_2(z2, z1_target=z1,
                                 compute_loss=self.training, **kwargs)
                wandb_loss = {"px1": out["losses"]}
            else:
                out = self.enc_3(z1, z1_target=z2,
                                 compute_loss=self.training, **kwargs)
                wandb_loss = {"px2": out["losses"]}

            if "current_step" in kwargs:
                import wandb
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)

            return out

        return {"tilde": z1}

    # ========================================================
    # Apply perturbation
    # ========================================================

    def forward_once_perturbed(self, a, v, **kwargs):

        if self.cls_type == "tf":
            z1 = a["nonaggr_features"]["combined"]
            z2 = v["nonaggr_features"]["combined"]
        else:
            z1 = a["features"]["combined"]
            z2 = v["features"]["combined"]

        pert_losses = {}

        if kwargs.get("perturb_x1", False):
            out = self._perturb_features(z1=z1, z2=z2, swap_direction=False, **kwargs)
            pert_z1 = out["tilde"].detach()
            pert_losses.update({"px1": out["losses"]})

        if kwargs.get("perturb_x2", False):
            out = self._perturb_features(z1=z1, z2=z2, swap_direction=True, **kwargs)
            pert_z2 = out["tilde"].detach()
            pert_losses.update({"px2": out["losses"]})

        return {
            "features": {
                "z1": z1,
                "z2": z2,
                "pert_z1": pert_z1 if "pert_z1" in locals() else None,
                "pert_z2": pert_z2 if "pert_z2" in locals() else None
            },
            "losses": pert_losses
        }

    # ========================================================
    # Main forward
    # ========================================================

    def forward(self, x, **kwargs):

        output = {"preds": {}, "features": {}, "losses": {}}

        a, v = self._get_features(x, **kwargs)
        output["features"]["z1"] = a["features"]["combined"]
        output["features"]["z2"] = v["features"]["combined"]

        if self.training:

            p1 = self.forward_once_perturbed(
                a, v, stochastic=True, perturb_x1=True, perturb_x2=False, **kwargs)
            output["losses"].update(p1["losses"])
            output["features"]["~z1"] = p1["features"]["pert_z1"]

            p2 = self.forward_once_perturbed(
                a, v, stochastic=True, perturb_x1=False, perturb_x2=True, **kwargs)
            output["losses"].update(p2["losses"])
            output["features"]["~z2"] = p2["features"]["pert_z2"]

        else:

            p1 = self.forward_once_perturbed(
                a, v, stochastic=False, perturb_x1=True, perturb_x2=False, **kwargs)
            p2 = self.forward_once_perturbed(
                a, v, stochastic=False, perturb_x1=False, perturb_x2=True, **kwargs)

            output["features"]["~z1"] = p1["features"]["pert_z1"]
            output["features"]["~z2"] = p2["features"]["pert_z2"]

        return output
class SimpleVAEZ2toZ1OneSide(nn.Module):
    """
    Minimal standalone VAE used as enc_2:
        - initialized with enc = enc_class(args=args, encs=[])
        - KL weight is called `lib`
        - no dependencies on external encoders
        - takes dim_in, dim_out, etc. from args
    """

    def __init__(self, args, encs=[]):
        super().__init__()

        self.args = args

        # read VAE hyperparameters from args
        dim_in = args.get("dim_in", args.get("latent_dim", 512))
        dim_out = args.get("dim_out", args.get("latent_dim", 512))
        hidden = args.get("hidden", 512)
        latent_dim = args.get("latent_dim_inner", 64)
        lib_max = args.get("lib_max", 1.0)

        self.latent_dim = latent_dim
        self.lib_max = lib_max

        # dynamic KL/IB weight
        self.register_buffer("lib", torch.tensor(0.0))

        # ------------------ Encoder ------------------
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        # ------------------ Decoder ------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_out)
        )

    # ------------------------------------------------
    # KL/IB Weight Setter
    # ------------------------------------------------
    def set_lib_weight(self, value):
        """Warm-up setter for lib."""
        value = max(0.0, min(self.lib_max, value))
        self.lib.fill_(value)

    # ------------------------------------------------
    # Forward pass
    # ------------------------------------------------
    def forward(self, z_in, z_target=None, compute_loss=True, **kwargs):

        # Encode
        h = self.encoder(z_in)
        mu = self.mu(h)
        logvar = self.logvar(h)

        # Sample latent
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_lat = mu + eps * std

        # Decode
        pred = self.decoder(z_lat)

        recon = F.mse_loss(pred, z_target, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon + self.lib * kl
        if self.args.freeze_encoder:
            return {
                "tilde": pred
            }
        if self.training:
            return {
                "tilde": pred,
                "losses": {
                    "total": total,
                    "recon": recon.detach(),
                    "kl": kl.detach(),
                    "lib": self.lib.item()
                }
            }
        else:
            return {
                "tilde": pred,
                "losses": {
                    "total": recon.detach()
                }
            }
class Model_GeneratorVAEOneSide(nn.Module):
    """
    One-sided VAE-based generator:
        swap_directions = True  → enc_2(z2) → z1
        swap_directions = False → enc_2(z1) → z2

    enc_2 is passed from config!
    """

    def __init__(self, args, encs):
        super().__init__()

        self.swap_directions = args.get("swap_directions", True)
        self.perturb_type = args.perturb
        dim_z = args.get("latent_dim", 512)

        # unimodal encoders
        self.enc_0 = encs[0]       # audio
        self.enc_1 = encs[1]       # video

        # --- VAE comes from config -------------------------
        self.enc_2 = encs[2]       # SimpleVAEZ2toZ1 from config
        # ----------------------------------------------------

    def _get_features(self, x, **kwargs):
        z1 = self.enc_0(x, **kwargs)["features"]["combined"]
        z2 = self.enc_1(x, **kwargs)["features"]["combined"]
        return z1, z2

    def _perturb(self, z1, z2):

        if self.swap_directions:
            out = self.enc_2(z2, z_target=z1, compute_loss=self.training)
        else:
            out = self.enc_2(z1, z_target=z2, compute_loss=self.training)

        return out["tilde"].detach(), out["losses"]

    def forward(self, x, **kwargs):
        z1, z2 = self._get_features(x, **kwargs)
        out = {"features": {"z1": z1, "z2": z2}, "losses": {}, "preds":{}}

        tilde, losses = self._perturb(z1, z2)
        out["features"]["~z"] = tilde
        out["losses"]["px"] = losses

        return out



class MCR_Syn_Linear(nn.Module):

    def __init__(self, args: dict, encs: list[nn.Module]):
        super().__init__()

        self.args: dict = args
        num_classes: int = args["num_classes"]
        d_model: int = args["d_model"]

        self.enc_0: nn.Module = encs[0]
        self.enc_1: nn.Module = encs[1]
        self.enc_2: nn.Module = encs[2]
        self.enc_3: nn.Module = encs[3]

        self.use_bias_infusion: bool = args.get("bias_infusion", {}).get("l", 0) != 0

        self.fc_0: nn.Linear = nn.Linear(d_model, num_classes, bias=False)
        self.fc_1: nn.Linear = nn.Linear(d_model, num_classes, bias=False)
        self.bias_lin: nn.Parameter = nn.Parameter(torch.zeros(num_classes))

        self.bias_infusion = args.get("bias_infusion", {})
        self.lib = self.bias_infusion.get("lib", 0) > 0
        self.l = self.bias_infusion.get("l", 0) > 0
        self.temperature = self.bias_infusion.get("temperature", 0.1)
        self.contr_coeff = self.bias_infusion.get("contr_coeff", 0.0)
        self.contr_type = self.bias_infusion.get("contr_type", "label")
        self.regby = self.bias_infusion.get("regby", "greedy")

        if self.lib > 0:
            self.fc_yz: nn.Sequential = nn.Sequential(
                nn.Linear(num_classes, d_model, bias=False),
                nn.ReLU(),
                nn.Linear(d_model, d_model * 2, bias=False),
            )

    def _get_features(self, x: torch.Tensor, **enc_kwargs):
        """
        Run both encoders and extract their combined feature tensors.

        Returns
        -------
        a_feat : torch.Tensor
            Features from enc_0, shape (B, d_model)
        v_feat : torch.Tensor
            Features from enc_1, shape (B, d_model)
        a_pred : torch.Tensor
            Unimodal predictions from enc_0, shape (B, num_class)
        v_pred : torch.Tensor
            Unimodal predictions from enc_1, shape (B, num_class)
        """
        out_a = self.enc_0(x, **enc_kwargs)
        out_v = self.enc_1(x, **enc_kwargs)

        a_feat = out_a["features"]["combined"]
        v_feat = out_v["features"]["combined"]
        a_pred = out_a["preds"]["combined"]
        v_pred = out_v["preds"]["combined"]

        return a_feat, v_feat, a_pred, v_pred

    def _linear_forward(
        self,
        a_feat: torch.Tensor,
        v_feat: torch.Tensor,
        detach_a: bool = False,
        detach_v: bool = False,
        skip_bias: bool = False,
    ) -> torch.Tensor:
        """
        Compute the linear fusion prediction:
            pred = a_feat * fc_0 + v_feat * fc_1 + bias_lin

        Parameters
        ----------
        a_feat, v_feat : torch.Tensor
            Feature matrices for each modality.
        detach_a, detach_v : bool
            If True, detach the corresponding features.
        skip_bias : bool
            If True, do not add the bias term.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        if detach_a:
            a_feat = a_feat.detach()
        if detach_v:
            v_feat = v_feat.detach()

        pred_a = self.fc_0(a_feat)
        pred_v = self.fc_1(v_feat)

        pred = pred_a + pred_v

        if not skip_bias:
            pred = pred + self.bias_lin

        return pred

    def _perturb_features(self, z1, z2, **kwargs):

        if self.perturb["type"] == "zeros":
            return {"tilde":torch.zeros_like(z1).detach()}
        elif self.perturb["type"] == "cmn": #conditional masked and noisy
            mask = (torch.rand_like(z1) < self.perturb.get("p",0.5)).float()  # partial mask
            scale = self.perturb.get("s", 0.1) * (1 + torch.tanh(z2.norm(dim=-1, keepdim=True) / self.perturb.get("c",1)))
            noise = torch.randn_like(z1) * scale
            tilde_z1 = mask * z1 + (1 - mask) * noise
            return {"tilde": tilde_z1.detach()}
        elif self.perturb["type"] == "gen" or  self.perturb["type"] == "gen_v2":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, current_step=kwargs.get("current_step", None))
                wandb_loss = {"px1": gen_out["losses"]} if "losses" in gen_out else {}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, current_step=kwargs.get("current_step", None))
                wandb_loss = {"px2": gen_out["losses"]} if "losses" in gen_out else {}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out
        elif self.perturb["type"] == "gennoise":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out

    def _build_shuffled_batches(
            self,
            a_feat: torch.Tensor,
            v_feat: torch.Tensor,
            pred: torch.Tensor,
            labels: torch.Tensor,
    ):
        """
        Build expanded shuffled and repeated batches.
        Shapes: (B * num_samples, d_model) or (B * num_samples,)
        """
        B = labels.size(0)
        num_samples = self.args["bias_infusion"].get("num_samples", 1)

        pert_out_1 = self._perturb_features(z1=a_feat, z2=v_feat, swap_direction=True)
        pert_out_2 = self._perturb_features(z1=a_feat, z2=v_feat, swap_direction=False)

        # Expanded shuffled features
        shuffled_a = pert_out_1["tilde"]
        shuffled_v = pert_out_2["tilde"]

        # Repeated originals
        repeated = torch.cat([torch.arange(B) for _ in range(num_samples)], dim=0)

        a_repeated = a_feat[repeated]
        v_repeated = v_feat[repeated]
        pred_repeated = pred[repeated]
        label_repeated = labels[repeated]

        return {
            "shuffled_a": shuffled_a,
            "shuffled_v": shuffled_v,
            "a_repeated": a_repeated,
            "v_repeated": v_repeated,
            "pred_repeated": pred_repeated,
            "label_repeated": label_repeated
        }

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, **enc_kwargs):
        """
        Forward pass combining encoders + linear fusion.
        Produces additional outputs when bias infusion is enabled.

        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        labels : torch.Tensor, optional
            Ground-truth labels for bias infusion.
        enc_kwargs : dict
            Additional arguments forwarded to enc_0 and enc_1.

        Returns
        -------
        dict
            Contains:
                "pred" : logits
                "features" : {"a": a_feat, "v": v_feat}
                "bias_infusion": {...}   # optional
        """
        a_feat, v_feat, a_pred, v_pred = self._get_features(x, **enc_kwargs)
        pred = self._linear_forward(a_feat, v_feat)

        out = {
            "preds": {"combined": pred,
                      "c": a_pred,
                      "g": v_pred
                      },
            "features": {
                "a": a_feat,
                "v": v_feat,
            },
        }

        if (
            self.training
            and self.l>0
            and labels is not None
        ):
            s = self._build_shuffled_batches(a_feat, v_feat, pred, labels)

            out["preds"].update({

                # --- shuffled A with repeated V ---
                "sa_detv": self._linear_forward(s["shuffled_a"], s["v_repeated"], detach_v=True, skip_bias=True),
                "sa_deta": self._linear_forward(s["shuffled_a"], s["v_repeated"], detach_a=True, skip_bias=True),
                "sa": self._linear_forward(s["shuffled_a"], s["v_repeated"], skip_bias=True),

                # --- repeated A with shuffled V ---
                "sv_detv": self._linear_forward(s["a_repeated"], s["shuffled_v"], detach_v=True, skip_bias=True),
                "sv_deta": self._linear_forward(s["a_repeated"], s["shuffled_v"], detach_a=True, skip_bias=True),
                "sv": self._linear_forward(s["a_repeated"], s["shuffled_v"], skip_bias=True),

                # --- repeated predictions and labels ---
                "pred_repeated": s["pred_repeated"],
                "label_repeated": s["label_repeated"],
                "label_shuffled": s["label_shuffled"],
            })

            out["losses"], out["permutation_importance"] = self._compute_loss(out, labels)

        return out

    def _apply_game(self, lmipds):
        """
        Compute LMIPD-based regularization according to the selected game setting.

        lmipds keys expected:
            enc0_sa_jsd, enc0_sv_jsd,
            enc1_sa_jsd, enc1_sv_jsd,
            all_sa_jsd,  all_sv_jsd
        """

        # JSD scores for each encoder under each perturbation
        enc0_sa = lmipds["enc0_sa_jsd"]  # encoder 0 under A-shuffle
        enc0_sv = lmipds["enc0_sv_jsd"]  # encoder 0 under V-shuffle
        enc1_sa = lmipds["enc1_sa_jsd"]  # encoder 1 under A-shuffle
        enc1_sv = lmipds["enc1_sv_jsd"]  # encoder 1 under V-shuffle

        if self.regby == "greedy":
            # Use encoder-specific JSDs
            reg0 = enc0_sa
            reg1 = enc1_sv
            total = self.l * (-enc0_sa + enc1_sa - enc1_sv + enc0_sv)

        elif self.regby == "ind":
            # Penalize each encoder individually
            reg0 = enc0_sa
            reg1 = enc1_sv
            total = self.l * (-(enc0_sa + enc1_sv))

        else:  # "colab"
            # Cooperative minimization using modality-aggregated JSDs
            reg0 = lmipds["all_sa_jsd"]
            reg1 = lmipds["all_sv_jsd"]
            total = self.l * (-(reg0 + reg1))

        return {
            "total_reg": total,
            "reg0": reg0,
            "reg1": reg1,
        }

    def _compute_contrastive_term(self, output, label):
        if not self.contr_coeff:
            return 0.0

        feat_a = output["features"]["a"]
        feat_v = output["features"]["v"]
        t = self.temperature

        if self.contr_type == "label":
            loss = nt_xent_loss(feat_a, feat_v, label=label, temperature=t)
        else:
            loss = nt_xent_loss(feat_a, feat_v, temperature=t)

        return self.contr_coeff * loss

    def _get_lmipds(self, output):
        P = output["preds"]
        device = self.agent.device

        def jsd(p, q):
            p = F.softmax(p, dim=1)
            q = F.softmax(q, dim=1)
            m = 0.5 * (p + q)
            return 0.5 * (
                    F.kl_div(m.log(), p, reduction="batchmean") +
                    F.kl_div(m.log(), q, reduction="batchmean")
            )

        def eval_side(side):
            if side == "sa":
                keys = ("sa_detv", "sa_deta", "sa")
            else:
                keys = ("sv_detv", "sv_deta", "sv")

            if all(k in P for k in keys):
                detv, deta, both = P[keys[0]], P[keys[1]], P[keys[2]]
                base = P["pred_repeated"]
                return {
                    "enc0": jsd(detv, base),
                    "enc1": jsd(deta, base),
                    "all": jsd(both, base),
                }
            else:
                zero = torch.tensor(0.0, device=device)
                return {"enc0": zero, "enc1": zero, "all": zero}

        sa = eval_side("sa")
        sv = eval_side("sv")

        base = P["pred_repeated"]
        base_acc = (base.argmax(1) == P["label_repeated"]).float().mean() + 1e-8
        acc_sa = (P["sa_detv"].argmax(1) == P["label_repeated"]).float().mean() if "sa_detv" in P else torch.tensor(0.0, device=device)
        acc_sv = (P["sv_deta"].argmax(1) == P["label_repeated"]).float().mean() if "sv_deta" in P else torch.tensor(0.0, device=device)

        permutation_importance = {
            "enc_0": acc_sv / base_acc,
            "enc_1": acc_sa / base_acc,
        }

        out = {
            "enc0_sa_jsd": sa["enc0"],
            "enc1_sa_jsd": sa["enc1"],
            "all_sa_jsd": sa["all"],

            "enc0_sv_jsd": sv["enc0"],
            "enc1_sv_jsd": sv["enc1"],
            "all_sv_jsd": sv["all"],
        }

        return out, permutation_importance

    def _reconstruction_loss(self, output):
        """
        Reconstruct features from predicted labels using fc_yz.
        Returns MSE loss or 0 if reconstruction is disabled.
        """
        if self.lib <=0.0:
            return torch.tensor(0.0, device=self.bias_lin.device)

        preds = output["preds"]["combined"].detach()              # B × num_classes
        a_feat = output["features"]["a"]                          # B × d_model
        v_feat = output["features"]["v"]                          # B × d_model
        true_feat = torch.cat([a_feat, v_feat], dim=1)            # B × (2*d_model)

        recon = self.fc_yz(preds)                                 # B × (2*d_model)

        return F.mse_loss(recon, true_feat)

    def _log_reg_terms(self, lmipds, lmipd_reg, contrastive_reg, reconstr_reg):
        try:
            if hasattr(wandb, "run") and wandb.run is not None:
                wandb.log(
                    {
                        "lmipds": lmipds,
                        "lmipd_reg": lmipd_reg,
                        "contrastive_reg": contrastive_reg,
                        "reconstruction_reg": reconstr_reg,
                    },
                    step=self.agent.logs["current_step"] + 1
                )
        except Exception:
            pass  # completely silent fail

    def _compute_loss(self, output, labels):
        """
        Compute:
            - LMIPD regularizer
            - Contrastive regularizer
            - Reconstruction regularizer
        plus logs each component.
        """

        lmipds, permutation_importance = self._get_lmipds(output)
        lmipd_reg = self._apply_game(lmipds)

        contrastive_reg = self._compute_contrastive_term(output, labels)

        reconstr_reg = self._reconstruction_loss(output)

        output_losses = {
            "lmipd": lmipd_reg["total_reg"],
            "contrastive": contrastive_reg,
            "reconstruction": reconstr_reg,
        }

        self._log_reg_terms(
            lmipds=lmipds,
            lmipd_reg=lmipd_reg,
            contrastive_reg=contrastive_reg,
            reconstr_reg=reconstr_reg
        )

        return output_losses, permutation_importance

class Fusion_Synprom_IB(nn.Module):
    def __init__(self, args, encs):
        super(Fusion_Synprom_IB, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        self.perturb = {"type": args.perturb}


        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        bias_infusion = args.get("bias_infusion",{})
        self.l = bias_infusion.get("l", 0)
        self.contrcoeff = bias_infusion.get("contrcoeff", False)



        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )
            self.mu_head = nn.Linear(d_model, num_classes)
            self.logvar_head = nn.Linear(d_model, num_classes)

        elif self.cls_type == "conformer":

            self.common_fc_1 =TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6, output_dim=d_model)

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )

        else:
            raise ValueError("Unknown cls_type")

        self.mu_head = nn.Linear(fc_inner, num_classes)
        self.logvar_head = nn.Linear(fc_inner, num_classes)

        # self.z1generator = Z1Generator_v2(d_model, d_model, hidden_dim=128, bottleneck_dim=128, variational=False)
        # self.z2generator = Z1Generator_v2(d_model, d_model, hidden_dim=128, bottleneck_dim=128, variational=False)


    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)

        return a, v

    def _forward_main(self, a, v, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1))
            feat = self.common_fc_2(aggr_feat)

        if self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], return_all=True)
            feat = self.common_fc_2(aggr_feat)

        pred_mu = self.mu_head(feat)
        pred_logvar = self.logvar_head(feat)
        if "stochastic" in kwargs and kwargs["stochastic"]:
            std = torch.exp(0.5 * pred_logvar)
            eps = torch.randn_like(std)
            pred = pred_mu + eps * std

            return pred, pred_mu, pred_logvar
        # else:
        #     std = torch.exp(0.5 * pred_logvar)
        #     eps = torch.randn_like(std)
        #     pred_mu = pred_mu + eps * std

        return pred_mu, aggr_feat

    def forward_once(self, x, **kwargs):

        a, v = self._get_features(x, **kwargs)

        if "perturb_x1" in kwargs and kwargs["perturb_x1"]:
            pert_out = self._perturb_features(z1=a["features"]["combined"], z2=v["features"]["combined"], swap_direction=True, **kwargs)
            a["features"]["combined"] = pert_out["tilde"].detach()
            # v["features"]["combined"] = v["features"]["combined"].detach()
            pert_losses = {"px1":pert_out["losses"]} if "losses" in pert_out else {}

        if "perturb_x2" in kwargs and kwargs["perturb_x2"]:
            pert_out = self._perturb_features(z1=v["features"]["combined"], z2=a["features"]["combined"], **kwargs)
            v["features"]["combined"] = pert_out["tilde"].detach()
            # a["features"]["combined"] = a["features"]["combined"].detach()
            pert_losses = {"px2":pert_out["losses"]} if "losses" in pert_out else {}



        if "stochastic" in kwargs and kwargs["stochastic"]:
            pred, mu, logvar = self._forward_main(a, v, **kwargs)
            pl = pert_losses if "pert_losses" in locals() else {}
            return self._kl_loss(mu, logvar), pl

        pred, feat = self._forward_main(a, v, **kwargs)

        output = {  "preds":{   "combined":pred,
                                "p1":a["preds"]["combined"],
                                "p2":v["preds"]["combined"]
                            },
                    "features": {"z1": a["features"]["combined"],
                                "z2": v["features"]["combined"]
                                 },
                    "losses": {}
                  }

        return output

    def _perturb_features(self, z1, z2, **kwargs):

        if self.perturb["type"] == "zeros":
            return {"tilde":torch.zeros_like(z1).detach()}
        elif self.perturb["type"] == "cmn": #conditional masked and noisy
            mask = (torch.rand_like(z1) < self.perturb.get("p",0.5)).float()  # partial mask
            scale = self.perturb.get("s", 0.1) * (1 + torch.tanh(z2.norm(dim=-1, keepdim=True) / self.perturb.get("c",1)))
            noise = torch.randn_like(z1) * scale
            tilde_z1 = mask * z1 + (1 - mask) * noise
            return {"tilde": tilde_z1.detach()}
        elif self.perturb["type"] == "gen" or  self.perturb["type"] == "gen_v2":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, current_step=kwargs.get("current_step", None))
                wandb_loss = {"px1": gen_out["losses"]} if "losses" in gen_out else {}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, current_step=kwargs.get("current_step", None))
                wandb_loss = {"px2": gen_out["losses"]} if "losses" in gen_out else {}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out
        elif self.perturb["type"] == "gennoise":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out


    def _kl_loss(self, mu_p, logvar_p):
        return 0.5 * torch.sum( torch.exp(logvar_p) + mu_p**2 - 1.0 - logvar_p, dim=1 ).mean()

    def forward(self, x, **kwargs):

        output = self.forward_once(x, **kwargs)

        if self.training:

            kl_1, pert_losses = self.forward_once( x, stochastic=True, perturb_x1=True, perturb_x2=False, **kwargs)
            output["losses"].update(pert_losses)

            kl_2, pert_losses = self.forward_once(x, stochastic=True, perturb_x1=False, perturb_x2=True, **kwargs)
            output["losses"].update(pert_losses)


            infonce_loss = nt_xent_loss(output["features"]["z1"], output["features"]["z2"], temperature=1.0)

            # supcon_loss = nt_xent_loss(output["features"]["c"], output["features"]["g"], label=label, temperature=self.temperature)

            wandb_loss = {"reg_loss": {"kl_1": kl_1, "kl_2": kl_2, "infonce": infonce_loss}}
            wandb.log(wandb_loss, step=kwargs["current_step"]+1)

            output["losses"].update({
                "sl_1": kl_1*self.l,
                "sl_2": kl_2*self.l,
                "infonce": infonce_loss*self.contrcoeff
            })
        else:
            output_z1 = self.forward_once( x, stochastic=False, perturb_x1=True, perturb_x2=False, **kwargs)
            output_z2 = self.forward_once( x, stochastic=False, perturb_x1=False, perturb_x2=True, **kwargs)
            output["features"]["~z1"] = output_z1["features"]["z1"]
            output["features"]["~z2"] = output_z2["features"]["z2"]


        return output

class Fusion_Synprom_IB_Dir(nn.Module):
    def __init__(self, args, encs):
        super(Fusion_Synprom_IB_Dir, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        self.perturb = {"type": args.perturb}

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        bias_infusion = args.get("bias_infusion", {})
        self.l = bias_infusion.get("l", 0)
        self.contrcoeff = bias_infusion.get("contrcoeff", False)

        # ---- feature fusion / common trunk ----
        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )

        elif self.cls_type == "conformer":

            self.common_fc_1 = TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6,
                output_dim=d_model
            )

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )

        else:
            raise ValueError("Unknown cls_type")

        # ---- Dirichlet head instead of Gaussian (mu, logvar) ----
        # Unconstrained outputs in ℝ^K → turned into α(x) > 0
        self.alpha_head = nn.Linear(fc_inner, num_classes)

        # Prior Dir(α⁰); default α⁰ = 1 for all classes (uniform Dirichlet)
        prior_conc = args.get("dir_prior_conc", 1.0)
        self.register_buffer(
            "alpha0",
            prior_conc * torch.ones(num_classes)
        )

    # ----------------------------------------------------------------------------------
    # Core subroutines
    # ----------------------------------------------------------------------------------

    def _get_features(self, x, **kwargs):
        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)
        return a, v

    def _forward_main(self, a, v, **kwargs):
        """
        Compute fused representation and Dirichlet parameters α(x).
        Returns:
            alpha: (B, K) > 0
            aggr_feat: fused feature (for logging / contrastive)
        """
        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(
                torch.cat(
                    (a["features"]["combined"], v["features"]["combined"]),
                    dim=1
                )
            )
            feat = self.common_fc_2(aggr_feat)

        elif self.cls_type == "conformer":
            # TF_Fusion returns (pred, aggr_feat, feat); we only need aggr_feat
            _, aggr_feat, _ = self.common_fc_1(
                [a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]],
                return_all=True
            )
            feat = self.common_fc_2(aggr_feat)

        else:
            raise ValueError("Unknown cls_type")

        # Dirichlet parameters α(x) > 0
        alpha_logits = self.alpha_head(feat)          # (B, K) in ℝ
        alpha = F.softplus(alpha_logits) + 1e-4       # (B, K) in ℝ_{>0}

        return alpha, aggr_feat

    # ----------------------------------------------------------------------------------
    # Forward once: used both for normal pass and KL computation
    # ----------------------------------------------------------------------------------

    def forward_once(self, x, **kwargs):

        a, v = self._get_features(x, **kwargs)

        # ----------------- perturb features if requested -----------------
        if "perturb_x1" in kwargs and kwargs["perturb_x1"]:
            pert_out = self._perturb_features(
                z1=a["features"]["combined"],
                z2=v["features"]["combined"],
                swap_direction=True,
                **kwargs
            )
            a["features"]["combined"] = pert_out["tilde"].detach()
            pert_losses = {"px1": pert_out["losses"]} if "losses" in pert_out else {}

        if "perturb_x2" in kwargs and kwargs["perturb_x2"]:
            pert_out = self._perturb_features(
                z1=v["features"]["combined"],
                z2=a["features"]["combined"],
                **kwargs
            )
            v["features"]["combined"] = pert_out["tilde"].detach()
            pert_losses = {"px2": pert_out["losses"]} if "losses" in pert_out else {}

        # ----------------- stochastic path: only compute KL -----------------
        if "stochastic" in kwargs and kwargs["stochastic"]:
            alpha, _ = self._forward_main(a, v, **kwargs)   # α(x)
            pl = pert_losses if "pert_losses" in locals() else {}
            return self._kl_loss(alpha), pl                # KL(Dir(α) || Dir(α0))

        # ----------------- deterministic path: use Dirichlet mean as prediction -----------------
        alpha, feat = self._forward_main(a, v, **kwargs)   # (B, K), α > 0

        # Dirichlet mean: E[p_k] = α_k / ∑_j α_j
        pred_mean = alpha / alpha.sum(dim=-1, keepdim=True)    # (B, K), on simplex

        output = {
            "preds": {
                "combined": pred_mean,
                "p1": a["preds"]["combined"],
                "p2": v["preds"]["combined"]
            },
            "features": {
                "z1": a["features"]["combined"],
                "z2": v["features"]["combined"]
            },
            "losses": {}
        }

        return output

    # ----------------------------------------------------------------------------------
    # Perturbation of features (unchanged)
    # ----------------------------------------------------------------------------------

    def _perturb_features(self, z1, z2, **kwargs):

        if self.perturb["type"] == "zeros":
            return {"tilde": torch.zeros_like(z1).detach()}

        elif self.perturb["type"] == "cmn":  # conditional masked and noisy
            mask = (torch.rand_like(z1) < self.perturb.get("p", 0.5)).float()  # partial mask
            scale = self.perturb.get("s", 0.1) * (
                1 + torch.tanh(z2.norm(dim=-1, keepdim=True) / self.perturb.get("c", 1))
            )
            noise = torch.randn_like(z1) * scale
            tilde_z1 = mask * z1 + (1 - mask) * noise
            return {"tilde": tilde_z1.detach()}

        elif self.perturb["type"] == "gen" or self.perturb["type"] == "gen_v2":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training,
                                     current_step=kwargs.get("current_step", None))
                wandb_loss = {"px1": gen_out["losses"]} if "losses" in gen_out else {}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training,
                                     current_step=kwargs.get("current_step", None))
                wandb_loss = {"px2": gen_out["losses"]} if "losses" in gen_out else {}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out

        elif self.perturb["type"] == "gennoise":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out

    # ----------------------------------------------------------------------------------
    # KL loss: KL( Dir(alpha) || Dir(alpha0) )
    # ----------------------------------------------------------------------------------

    def _kl_loss(self, alpha):
        """
        KL( Dir(alpha) || Dir(alpha0) ), averaged over batch.

        alpha: (B, K)  posterior concentration parameters α(x)
        alpha0: (K,)   prior concentration parameters α⁰ (buffer)
        """
        alpha0 = self.alpha0                 # (K,)
        sum_alpha = alpha.sum(dim=-1)        # (B,)
        sum_alpha0 = alpha0.sum()            # scalar

        # log B(α0) - log B(α)
        # log B(α) = sum lgamma(α_k) - lgamma(sum α_k)
        log_B_alpha0 = torch.lgamma(alpha0).sum() - torch.lgamma(sum_alpha0)
        log_B_alpha = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(sum_alpha)
        term1 = log_B_alpha0 - log_B_alpha          # (B,)

        # sum_k (α_k - α0_k) [ψ(α_k) - ψ(sum α)]
        digamma_alpha = torch.digamma(alpha)                       # (B, K)
        digamma_sum_alpha = torch.digamma(sum_alpha).unsqueeze(-1) # (B, 1)
        term2 = ((alpha - alpha0) * (digamma_alpha - digamma_sum_alpha)).sum(dim=-1)

        kl = term1 + term2            # (B,)
        return kl.mean()


    def forward(self, x, **kwargs):

        output = self.forward_once(x, **kwargs)

        if self.training:

            # two stochastic passes with perturbations, just to compute KL + perturb losses
            kl_1, pert_losses = self.forward_once(
                x,
                stochastic=True,
                perturb_x1=True,
                perturb_x2=False,
                **kwargs
            )
            output["losses"].update(pert_losses)

            kl_2, pert_losses = self.forward_once(
                x,
                stochastic=True,
                perturb_x1=False,
                perturb_x2=True,
                **kwargs
            )
            output["losses"].update(pert_losses)

            # contrastive loss on z1, z2 (unchanged)
            infonce_loss = nt_xent_loss(
                output["features"]["z1"],
                output["features"]["z2"],
                temperature=1.0
            )

            wandb_loss = {
                "reg_loss": {
                    "kl_1": kl_1,
                    "kl_2": kl_2,
                    "infonce": infonce_loss
                }
            }
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            else:
                wandb.log(wandb_loss)

            # add scaled regularizers
            output["losses"].update({
                "sl_1": kl_1 * self.l,
                "sl_2": kl_2 * self.l,
                "infonce": infonce_loss * self.contrcoeff
            })

        else:
            # for eval: compute perturbed z1, z2 once (deterministic Dirichlet mean)
            output_z1 = self.forward_once(
                x,
                stochastic=False,
                perturb_x1=True,
                perturb_x2=False,
                **kwargs
            )
            output_z2 = self.forward_once(
                x,
                stochastic=False,
                perturb_x1=False,
                perturb_x2=True,
                **kwargs
            )
            output["features"]["~z1"] = output_z1["features"]["z1"]
            output["features"]["~z2"] = output_z2["features"]["z2"]

        return output




class Model_Generator(nn.Module):
    def __init__(self, args, encs):
        super(Model_Generator, self).__init__()

        self.args = args
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        self.cls_type = args.get("cls_type", "tf")
        self.perturb = {"type": args.perturb}

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]


    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)

        return a, v

    def _forward_main(self, a, v, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1))
            feat = self.common_fc_2(aggr_feat)

        if self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], return_all=True)
            feat = self.common_fc_2(aggr_feat)

        pred_mu = self.mu_head(feat)
        pred_logvar = self.logvar_head(feat)
        if "stochastic" in kwargs and kwargs["stochastic"]:
            std = torch.exp(0.5 * pred_logvar)
            eps = torch.randn_like(std)
            pred = pred_mu + eps * std

            return pred, pred_mu, pred_logvar
        # else:
        #     std = torch.exp(0.5 * pred_logvar)
        #     eps = torch.randn_like(std)
        #     pred_mu = pred_mu + eps * std

        return pred_mu, aggr_feat

    def forward_once(self, x, **kwargs):

        a, v = self._get_features(x, **kwargs)

        if "perturb_x1" in kwargs and kwargs["perturb_x1"]:
            # pert_out = self._perturb_features(z1=a["features"]["combined"], z2=v["features"]["combined"], swap_direction=True, **kwargs)
            pert_out = self._perturb_features(z1=a["nonaggr_features"]["combined"], z2=v["nonaggr_features"]["combined"], swap_direction=True, **kwargs)
            # a["features"]["combined"] = pert_out["tilde"].detach()
            a["nonaggr_features"]["combined"] = pert_out["tilde"].detach()
            # v["features"]["combined"] = v["features"]["combined"].detach()
            pert_losses = {"px1":pert_out["losses"]}

        if "perturb_x2" in kwargs and kwargs["perturb_x2"]:
            # pert_out = self._perturb_features(z1=v["features"]["combined"], z2=a["features"]["combined"], **kwargs)
            pert_out = self._perturb_features(z1=v["nonaggr_features"]["combined"], z2=a["nonaggr_features"]["combined"], **kwargs)
            # v["features"]["combined"] = pert_out["tilde"].detach()
            v["nonaggr_features"]["combined"] = pert_out["tilde"].detach()
            # a["features"]["combined"] = a["features"]["combined"].detach()
            pert_losses = {"px2":pert_out["losses"]}



        if "stochastic" in kwargs and kwargs["stochastic"]:
            pl = pert_losses if "pert_losses" in locals() else {}
            return pl


        output = {  "preds":{ },
                    "features": {"z1": a["features"]["combined"],
                                "z2": v["features"]["combined"]
                                 },
                    "losses": {}
                  }

        return output

    def _perturb_features(self, z1, z2, **kwargs):

        if self.perturb["type"] == "zeros":
            return {"tilde":torch.zeros_like(z1).detach()}
        elif self.perturb["type"] == "cmn": #conditional masked and noisy
            mask = (torch.rand_like(z1) < self.perturb.get("p",0.5)).float()  # partial mask
            scale = self.perturb.get("s", 0.1) * (1 + torch.tanh(z2.norm(dim=-1, keepdim=True) / self.perturb.get("c",1)))
            noise = torch.randn_like(z1) * scale
            tilde_z1 = mask * z1 + (1 - mask) * noise
            return {"tilde": tilde_z1.detach()}
        elif self.perturb["type"] == "gen" or  self.perturb["type"] == "gen_v2":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, **kwargs)
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, **kwargs)
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out
        elif self.perturb["type"] == "gennoise":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out

        # noise = torch.randn_like(feat) * 0.1
        # return feat + noise

    def _kl_loss(self, mu_p, logvar_p):
        return 0.5 * torch.sum( torch.exp(logvar_p) + mu_p**2 - 1.0 - logvar_p, dim=1 ).mean()

    def forward_once_perturbed(self, a, v, **kwargs):
        if self.cls_type == "tf":
            z1, z2 = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
        elif self.cls_type == "mlp":
            z1, z2 = a["features"]["combined"], v["features"]["combined"]

        if "perturb_x1" in kwargs and kwargs["perturb_x1"]:
            pert_out = self._perturb_features(z1=z1, z2=z2, swap_direction=False, **kwargs)
            pert_z1 = pert_out["tilde"].detach()
            pert_losses = {"px1":pert_out["losses"]}

        if "perturb_x2" in kwargs and kwargs["perturb_x2"]:
            pert_out = self._perturb_features(z1=z1, z2=z2, swap_direction=True,  **kwargs)
            pert_z2 = pert_out["tilde"].detach()
            pert_losses = {"px2":pert_out["losses"]}

        output = {
                    "features": {"z1": z1,
                                "z2": z2,
                                 "pert_z1": pert_z1 if "pert_z1" in locals() else None,
                                "pert_z2": pert_z2 if "pert_z2" in locals() else None
                                 },
                    "losses": pert_losses
                  }

        return output

    def forward(self, x, **kwargs):

        output = {  "preds":{},
                    "features": {},
                    "losses": {}
                  }

        a, v = self._get_features(x, **kwargs)

        output["features"]["z1"] = a["features"]["combined"]
        output["features"]["z2"] = v["features"]["combined"]

        if self.training:
            pert_losses1 = self.forward_once_perturbed(a, v , stochastic=True, perturb_x1=True, perturb_x2=False, **kwargs)
            output["losses"].update(pert_losses1["losses"])

            pert_losses2 = self.forward_once_perturbed(a, v , stochastic=True, perturb_x1=False, perturb_x2=True, **kwargs)
            output["losses"].update(pert_losses2["losses"])

            output["features"]["~z1"] = pert_losses1["features"]["pert_z1"]
            output["features"]["~z2"] = pert_losses2["features"]["pert_z2"]

        else:
            output_z1 = self.forward_once_perturbed( a, v , stochastic=False, perturb_x1=True, perturb_x2=False, **kwargs)
            output_z2 = self.forward_once_perturbed( a, v , stochastic=False, perturb_x1=False, perturb_x2=True, **kwargs)

            output["features"]["~z1"] = output_z1["features"]["pert_z1"]
            output["features"]["~z2"] = output_z2["features"]["pert_z2"]


        return output

class Fusion_Synprom_IB_gen(nn.Module):
    def __init__(self, args, encs):
        super(Fusion_Synprom_IB_gen, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)

        self.num_classes = args.num_classes
        self.perturb = {"type": args.get("perturb", None)}


        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]
        self.enc_3 = encs[3]

        bias_infusion = args.get("bias_infusion",{})
        self.l = bias_infusion.get("l", 0)
        self.contrcoeff = bias_infusion.get("contrcoeff", False)



        if self.cls_type == "mlp":
            self.common_fc_1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model))

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )
            self.mu_head = nn.Linear(d_model, num_classes)
            self.logvar_head = nn.Linear(d_model, num_classes)

        elif self.cls_type == "conformer":

            self.common_fc_1 =TF_Fusion(
                input_dim=d_model,
                dim=d_model,
                layers=6, output_dim=d_model)

            self.common_fc_2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, fc_inner)
            )

        else:
            raise ValueError("Unknown cls_type")

        self.mu_head = nn.Linear(fc_inner, num_classes)
        self.logvar_head = nn.Linear(fc_inner, num_classes)

        # self.z1generator = Z1Generator_v2(d_model, d_model, hidden_dim=128, bottleneck_dim=128, variational=False)
        # self.z2generator = Z1Generator_v2(d_model, d_model, hidden_dim=128, bottleneck_dim=128, variational=False)


    def _get_features(self, x, **kwargs):
        a = self.enc_0(x, **kwargs)
        v = self.enc_1(x, **kwargs)

        return a, v

    def _forward_main(self, a, v, **kwargs):

        if self.cls_type == "mlp":
            aggr_feat = self.common_fc_1(torch.cat((a["features"]["combined"], v["features"]["combined"]), dim=1))
            feat = self.common_fc_2(aggr_feat)

        if self.cls_type == "conformer":
            pred, aggr_feat, feat = self.common_fc_1([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]], return_all=True)
            feat = self.common_fc_2(aggr_feat)

        pred_mu = self.mu_head(feat)
        pred_logvar = self.logvar_head(feat)
        if "stochastic" in kwargs and kwargs["stochastic"]:
            std = torch.exp(0.5 * pred_logvar)
            eps = torch.randn_like(std)
            pred = pred_mu + eps * std

            return pred, pred_mu, pred_logvar
        # else:
        #     std = torch.exp(0.5 * pred_logvar)
        #     eps = torch.randn_like(std)
        #     pred_mu = pred_mu + eps * std

        return pred_mu, aggr_feat

    def forward_once(self, x, **kwargs):

        a, v = self._get_features(x, **kwargs)
        z1 = a["features"]["combined"]
        z2 = v["features"]["combined"]

        if "perturb_x1" in kwargs and kwargs["perturb_x1"]:
            pert_out = self._perturb_features(z1=z1, z2=z2, swap_direction=False, **kwargs)
            z1 = pert_out["tilde"].detach()
            pert_losses = {"px1":pert_out["losses"]}

        if "perturb_x2" in kwargs and kwargs["perturb_x2"]:
            pert_out = self._perturb_features(z1=z1, z2=z2, swap_direction=True,  **kwargs)
            z2 = pert_out["tilde"].detach()
            pert_losses = {"px2":pert_out["losses"]}



        if "stochastic" in kwargs and kwargs["stochastic"]:
            pred, mu, logvar = self._forward_main(a, v, **kwargs)
            pl = pert_losses if "pert_losses" in locals() else {}
            return self._kl_loss(mu, logvar), pl

        pred, feat = self._forward_main(a, v, **kwargs)

        output = {  "preds":{   "combined":pred,
                                "p1":a["preds"]["combined"],
                                "p2":v["preds"]["combined"]
                            },
                    "features": {"z1": z1,
                                "z2": z2
                                 },
                    "losses": {}
                  }

        return output

    def forward_once_perturbed(self, z1, z2, **kwargs):

        if "perturb_x1" in kwargs and kwargs["perturb_x1"]:
            pert_out = self._perturb_features(z1=z1, z2=z2, swap_direction=False, **kwargs)
            pert_z1 = pert_out["tilde"].detach()
            pert_losses = {"px1":pert_out["losses"]}

        if "perturb_x2" in kwargs and kwargs["perturb_x2"]:
            pert_out = self._perturb_features(z1=z1, z2=z2, swap_direction=True,  **kwargs)
            pert_z2 = pert_out["tilde"].detach()
            pert_losses = {"px2":pert_out["losses"]}

        if "stochastic" in kwargs and kwargs["stochastic"]:
            pred, mu, logvar = self._forward_main(a, v, **kwargs)
            pl = pert_losses if "pert_losses" in locals() else {}
            return self._kl_loss(mu, logvar), pl

        output = {
                    "features": {"z1": z1,
                                "z2": z2,
                                 "pert_z1": pert_z1 if "pert_z1" in locals() else None,
                                "pert_z2": pert_z2 if "pert_z2" in locals() else None
                                 },
                    "losses": {}
                  }

        return output

    def _perturb_features(self, z1, z2, **kwargs):

        if self.perturb["type"] == "zeros":
            return {"tilde":torch.zeros_like(z1).detach()}
        elif self.perturb["type"] == "cmn": #conditional masked and noisy
            mask = (torch.rand_like(z1) < self.perturb.get("p",0.5)).float()  # partial mask
            scale = self.perturb.get("s", 0.1) * (1 + torch.tanh(z2.norm(dim=-1, keepdim=True) / self.perturb.get("c",1)))
            noise = torch.randn_like(z1) * scale
            tilde_z1 = mask * z1 + (1 - mask) * noise
            return {"tilde": tilde_z1.detach()}
        elif self.perturb["type"] == "gen" or  self.perturb["type"] == "gen_v2":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training)
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training)
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out
        elif self.perturb["type"] == "gennoise":
            if kwargs.get("swap_direction", False):
                gen_out = self.enc_2(z1, z2, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px1": gen_out["losses"]}
            else:
                gen_out = self.enc_3(z2, z1, compute_loss=self.training, add_noise=True)
                wandb_loss = {"px2": gen_out["losses"]}
            if "current_step" in kwargs:
                wandb.log(wandb_loss, step=kwargs["current_step"] + 1)
            return gen_out

        # noise = torch.randn_like(feat) * 0.1
        # return feat + noise

    def _kl_loss(self, mu_p, logvar_p):
        return 0.5 * torch.sum( torch.exp(logvar_p) + mu_p**2 - 1.0 - logvar_p, dim=1 ).mean()

    def forward(self, x, **kwargs):

        output = self.forward_once(x, **kwargs)

        if self.training:

            kl_1, pert_losses = self.forward_once( x, stochastic=True, perturb_x1=True, perturb_x2=False, **kwargs)
            output["losses"].update(pert_losses)

            kl_2, pert_losses = self.forward_once(x, stochastic=True, perturb_x1=False, perturb_x2=True, **kwargs)
            output["losses"].update(pert_losses)


            infonce_loss = nt_xent_loss(output["features"]["z1"], output["features"]["z2"], temperature=1.0)

            # supcon_loss = nt_xent_loss(output["features"]["c"], output["features"]["g"], label=label, temperature=self.temperature)

            wandb_loss = {"reg_loss": {"kl_1": kl_1, "kl_2": kl_2, "infonce": infonce_loss}}
            wandb.log(wandb_loss, step=kwargs["current_step"]+1)

            output["losses"].update({
                "sl_1": kl_1*self.l,
                "sl_2": kl_2*self.l,
                "infonce": infonce_loss*self.contrcoeff
            })
        else:
            output_z1 = self.forward_once_perturbed( x, stochastic=False, perturb_x1=True, perturb_x2=False, **kwargs)
            output_z2 = self.forward_once_perturbed( x, stochastic=False, perturb_x1=False, perturb_x2=True, **kwargs)
            output["features"]["~z1"] = output_z1["features"]["pert_z1"]
            output["features"]["~z2"] = output_z2["features"]["pert_z2"]


        return output

class XOR_Fusion_Synprom_perf(nn.Module):
    def __init__(self, args, encs):
        super(XOR_Fusion_Synprom_perf, self).__init__()

        self.args = args
        self.num_classes = args.num_classes
        d_model = args.d_model

        bias_infusion = args.get("bias_infusion",{})
        self.l = bias_infusion.get("l", 0)

        self.f_a = nn.Linear(16, d_model, bias=True)
        self.f_b = nn.Linear(16, d_model, bias=True)
        self.f_c = nn.Linear(16, d_model, bias=True)
        # self.f_d = nn.Linear(1, d_model, bias=True)
        # self.f_e = nn.Linear(1, d_model, bias=True)
        # self.f_f = nn.Linear(1, d_model, bias=True)
        # self.f_g = nn.Linear(1, d_model, bias=True)

        self.fc = nn.Sequential(
            nn.ReLU(),
            # nn.BatchNorm1d(d_model*7),
            nn.Linear(d_model*3, d_model, bias=True),
            nn.ReLU(),
            # nn.BatchNorm1d(d_model),
            # nn.Linear(d_model, d_model, bias=True),
            # nn.ReLU(),
            nn.Linear(d_model, self.num_classes, bias=True))

    def forward(self, x, **kwargs):

        feat_a = self.f_a(x[0])
        feat_b = self.f_b(x[1])
        feat_c = self.f_c(x[2])
        # feat_d = self.f_d(x[3])
        # feat_e = self.f_e(x[4])
        # feat_f = self.f_f(x[5])
        # feat_g = self.f_g(x[6])

        # [feat_a, feat_b, feat_c, feat_d, feat_e, feat_f, feat_g] = [F.normalize(v, p=2.0, dim=1) for v in [feat_a, feat_b, feat_c, feat_d, feat_e, feat_f, feat_g]]

        joint_feat = torch.concat([feat_a, feat_b, feat_c], dim=1)
        # joint_feat = torch.concat([feat_a, feat_b, feat_c, feat_d, feat_e, feat_f, feat_g], dim=1)
        pred = self.fc(joint_feat)

        output = {  "preds":{
                        "combined":pred
                            },
                    "features": {
                        "combined": torch.Tensor([1])
                    }
                  }

        if self.training:
            ground_truth_labels = kwargs["label"]

            #softmax

            p_multi = F.softmax(pred, dim=1).gather(1, ground_truth_labels.view(-1, 1))
            p_uni1 = F.softmax(pred_a, dim=1).gather(1, ground_truth_labels.view(-1, 1)).detach()
            p_uni2 = F.softmax(pred_v, dim=1).gather(1, ground_truth_labels.view(-1, 1)).detach()

            margin = 0

            loss_synergy1 = torch.mean(p_uni1 - p_multi + margin)+1
            loss_synergy2 = torch.mean(p_uni2 - p_multi + margin)+1

            wandb_loss = {"reg_loss": {"loss_synergy1": loss_synergy1, "loss_synergy2": loss_synergy2}}
            wandb.log(wandb_loss, step=kwargs["current_step"]+1)

            synergy_loss = loss_synergy1 + loss_synergy2
            output["losses"]={
                "sl": synergy_loss*self.l,
            }


        return output

#
# class TriModalFusionClassifier(nn.Module):
#     def __init__(self, args, encs=None):
#         super(TriModalFusionClassifier, self).__init__()
#
#         self.args = args
#         self.num_classes = args.num_classes
#         d_model = args.d_model
#         bias_infusion = args.get("bias_infusion",{})
#         self.l = bias_infusion.get("l", 0)
#         self.regby = bias_infusion.get("regby", None)
#
#         self.f_a = nn.Sequential(
#             nn.Linear(16, d_model),
#             # nn.ReLU(),
#             # nn.Linear(d_model, d_model)
#             # nn.LayerNorm(d_model)
#         )
#         self.f_b = nn.Sequential(
#             nn.Linear(16, d_model),
#             # nn.ReLU(),
#             # nn.Linear(d_model, d_model)
#             # nn.LayerNorm(d_model)
#         )
#         self.f_c = nn.Sequential(
#             nn.Linear(16, d_model),
#             # nn.ReLU(),
#             # nn.Linear(d_model, d_model)
#             # nn.LayerNorm(d_model)
#         )
#
#         # Combined + interactions → total 6 * d_model
#         fusion_in = d_model * 3
#         self.fc = nn.Sequential(
#             nn.Linear(fusion_in, d_model),
#             nn.ReLU(),
#             # nn.LayerNorm(d_model * 2),
#             # nn.Linear(d_model * 2, d_model),
#             # nn.ReLU(),
#             nn.Linear(d_model, self.num_classes)
#         )
#
#         # Optional unimodal heads (for auxiliary losses)
#         self.fc_a = nn.Linear(d_model, self.num_classes)
#         self.fc_b = nn.Linear(d_model, self.num_classes)
#         self.fc_c = nn.Linear(d_model, self.num_classes)
#
#     def forward(self, x, **kwargs):
#
#         # corr = torch.corrcoef(torch.stack([x[0][:,-1], kwargs["label"]]))[0, 1]
#         # print("Corr:", corr)
#
#         feat_a = self.f_a(x[0])
#         feat_b = self.f_b(x[1])
#         feat_c = self.f_c(x[2])
#
#         # Normalize (optional)
#         # feat_a, feat_b, feat_c = [F.normalize(f, p=2, dim=1) for f in [feat_a, feat_b, feat_c]]
#
#
#         # Fusion
#         joint_feat = torch.cat([feat_a, feat_b, feat_c], dim=1)
#         pred = self.fc(joint_feat)
#
#         # Unimodal predictions
#         pred_a = self.fc_a(feat_a)
#         pred_b = self.fc_b(feat_b)
#         pred_c = self.fc_c(feat_c)
#
#         output = {}
#         if self.training and "label" in kwargs:
#             labels = kwargs["label"]
#
#             if self.regby == "cond":
#                 loss_ab = conditional_alignment_loss(feat_a, feat_b, labels)
#                 loss_bc = conditional_alignment_loss(feat_b, feat_c, labels)
#                 loss_ca = conditional_alignment_loss(feat_c, feat_a, labels)
#                 output["losses"] = {"synergy": self.l * (loss_ab + loss_bc + loss_ca) / 3}
#             elif self.regby == "perf":
#                 synergy_conf_loss = synergy_confidence_loss(pred, [pred_a, pred_b, pred_c], labels)
#                 output["losses"] = {"synergy": self.l * synergy_conf_loss}
#
#         output["preds"]={"combined": pred, "a": pred_a, "b": pred_b, "c": pred_c}
#         output["features"]={"combined": joint_feat}
#         return output


class TriModalFusionClassifier(nn.Module):
    def __init__(self, args, encs, **kwargs):
        super().__init__()
        self.args = args
        self.num_classes = self.args.num_classes
        d_model = self.args.d_model
        self.f_a = nn.Linear(16, d_model)
        self.f_b = nn.Linear(16, d_model)
        self.f_c = nn.Linear(16, d_model)
        self.fc  = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.num_classes)
        )

    def forward(self, x, **kwargs):
        feat_a = self.f_a(x[0])
        feat_b = self.f_b(x[1])
        feat_c = self.f_c(x[2])
        joint_feat = torch.cat([feat_a, feat_b, feat_c], dim=1)
        pred = self.fc(joint_feat)

        output = {}
        output["preds"] = {"combined": pred}
        output["features"] = {"combined": joint_feat}
        return output


def conditional_alignment_loss( feat1, feat2, labels, temperature=0.1):
        """
        InfoNCE-style loss that encourages alignment between feat1 and feat2
        for samples sharing the same label.
        """
        # Normalize features
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        # Compute similarity matrix
        sim = torch.matmul(feat1, feat2.T) / temperature  # [N, N]

        # Build mask for same labels
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()  # [N, N]
        mask.fill_diagonal_(0)  # remove self-similarity

        # Log-softmax over similarities
        log_sim = F.log_softmax(sim, dim=1)

        # Positive pairs = same label
        pos = (log_sim * mask).sum(1) / mask.sum(1).clamp(min=1)

        # Take mean over batch
        loss = -pos.mean()
        return loss
def synergy_confidence_loss(pred_fusion, unimodal_preds, labels, margin=0.0):
    labels = labels.view(-1, 1)
    p_fusion = F.softmax(pred_fusion, dim=1).gather(1, labels)
    p_unis = [F.softmax(p, dim=1).gather(1, labels).detach() for p in unimodal_preds]
    p_uni_mean = torch.stack(p_unis, dim=0).mean(0)
    # Penalize only when fusion < unimodal - margin
    loss = torch.mean(F.relu(p_uni_mean - p_fusion + margin))
    return loss
#
# class TriModalFusionClassifier(nn.Module):
#     def __init__(self, args, encs=None):
#         super(TriModalFusionClassifier, self).__init__()
#
#         self.args = args
#         self.num_classes = args.num_classes
#         d_model = args.d_model
#
#         # Optional synergy regularization
#         bias_infusion = args.get("bias_infusion", {})
#         self.l = bias_infusion.get("l", 0)
#
#         # Modality-specific encoders
#         self.f_a = nn.Sequential(
#             nn.Linear(16, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )
#         self.f_b = nn.Sequential(
#             nn.Linear(16, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )
#         self.f_c = nn.Sequential(
#             nn.Linear(16, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model)
#         )
#
#         # Fusion network with cross-modal interactions
#         self.fc = nn.Sequential(
#             nn.Linear(d_model * 3, d_model * 2),
#             nn.ReLU(),
#             nn.LayerNorm(d_model * 2),
#             nn.Linear(d_model * 2, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, self.num_classes)
#         )
#
#         self.fc_1 = nn.Sequential(
#             nn.Linear(d_model, d_model * 2),
#             nn.ReLU(),
#             nn.LayerNorm(d_model * 2),
#             nn.Linear(d_model * 2, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, self.num_classes)
#         )
#         self.fc_2 = nn.Sequential(
#             nn.Linear(d_model, d_model * 2),
#             nn.ReLU(),
#             nn.LayerNorm(d_model * 2),
#             nn.Linear(d_model * 2, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, self.num_classes)
#         )
#         self.fc_3 = nn.Sequential(
#             nn.Linear(d_model, d_model * 2),
#             nn.ReLU(),
#             nn.LayerNorm(d_model * 2),
#             nn.Linear(d_model * 2, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, self.num_classes)
#         )
#
#
#     def forward(self, x, **kwargs):
#         # Encode each modality
#         feat_a = self.f_a(x[0])
#         feat_b = self.f_b(x[1])
#         feat_c = self.f_c(x[2])
#
#         # Optional: normalize features (helps synergy training)
#         feat_a, feat_b, feat_c = [F.normalize(f, p=2, dim=1) for f in [feat_a, feat_b, feat_c]]
#         joint_feat = torch.cat([feat_a, feat_b, feat_c], dim=1)
#         pred = self.fc(joint_feat)
#
#         pred_a = self.fc_1(feat_a)
#         pred_b = self.fc_2(feat_b)
#         pred_c = self.fc_3(feat_c)
#
#
#
#         output = {
#             "preds": {"combined": pred,
#                      "a": pred_a, "b": pred_b, "c": pred_c},
#             "features": {"combined": joint_feat}  # optionally return for synergy loss
#         }
#
#         return output


class Encoders(nn.Module):
    def __init__(self, d, logit_scale_init):
        super().__init__()
        self.f_a = nn.Linear(1, d, bias=True)
        self.f_b = nn.Linear(1, d, bias=True)
        self.f_c = nn.Linear(1, d, bias=True)
        self.f_d = nn.Linear(1, d, bias=True)
        self.f_e = nn.Linear(1, d, bias=True)
        self.f_f = nn.Linear(1, d, bias=True)
        self.f_g = nn.Linear(1, d, bias=True)
        self.f_h = nn.Linear(1, d, bias=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

    def forward(self, inputs):
        r_a = self.f_a(inputs[0])
        r_b = self.f_b(inputs[1])
        r_c = self.f_c(inputs[2])
        r_d = self.f_d(inputs[3])
        r_e = self.f_e(inputs[4])
        r_f = self.f_f(inputs[5])
        r_g = self.f_g(inputs[6])
        r_h = self.f_h(inputs[7])
        return r_a, r_b, r_c, r_d, r_e, r_f, r_g, r_h, self.logit_scale.exp()
