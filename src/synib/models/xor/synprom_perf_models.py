import torch
import torch.nn as nn
import torch.nn.functional as F
from synib.models.VAVL_git.VAVL.conformer.model import Conformer

# ----------------- Your fusion backbone (unchanged) -----------------
class TF_Fusion(nn.Module):
    def __init__(self, input_dim, dim, layers, output_dim):
        super().__init__()
        self.common_net = Conformer(
            input_dim=input_dim, encoder_dim=dim, num_encoder_layers=layers
        )
        self.cls_token  = nn.Parameter(torch.randn(1, 1, dim))
        self.mod_0_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.mod_1_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.mod_2_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.common_fc = nn.Linear(dim, output_dim)

    def forward(self, x, **kwargs):
        x_0 = x[0].permute(0, 2, 1)  # [B,T,d]
        x_1 = x[1].permute(0, 2, 1)
        x_0 = self.mod_0_token.repeat(x_0.shape[0], x_0.shape[1], 1) + x_0
        x_1 = self.mod_1_token.repeat(x_1.shape[0], x_1.shape[1], 1) + x_1
        xlist = [x_0, x_1]
        if len(x) > 2:
            x_2 = x[2].permute(0, 2, 1)
            x_2 = self.mod_2_token.repeat(x_2.shape[0], x_2.shape[1], 1) + x_2
            xlist.append(x_2)
        if kwargs.get("detach_a", False): xlist[0] = xlist[0].detach()
        if kwargs.get("detach_v", False): xlist[1] = xlist[1].detach()

        feat_mm = torch.cat([xi for xi in xlist], dim=1)
        feat_mm = torch.cat([self.cls_token.repeat(feat_mm.shape[0], 1, 1), feat_mm], dim=1)
        feat_mm = self.common_net(feat_mm)
        aggr_feat_mm = feat_mm[:, 0]
        pred = self.common_fc(aggr_feat_mm)
        return (pred, aggr_feat_mm, feat_mm) if kwargs.get("return_all", False) else pred


# ----------------- ULTRA-LIGHT complementarity modules -----------------
class LiteTransformerBaseline(nn.Module):
    """
    q_eta(y | Zk): tiny TransformerEncoder on token features (B,T,c) -> pooled -> logits.
    """
    def __init__(self, comp_dim, num_classes, nhead=2, num_layers=1, ff_mult=2, dropout=0.1, share_layer=True):
        super().__init__()
        ff_dim = ff_mult * comp_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=comp_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        # Parameter tying: reuse the same layer L times if share_layer=True
        if share_layer and num_layers > 1:
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.num_layers = num_layers
            self.shared_layer = enc_layer
        else:
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.num_layers = None
            self.shared_layer = None
        self.ln = nn.LayerNorm(comp_dim)
        self.cls = nn.Linear(comp_dim, num_classes)

    def forward(self, tokens, return_features=False):  # [B,T,c]
        if self.shared_layer is not None:
            h = tokens
            for _ in range(self.num_layers):
                h = self.enc.layers[0](h)  # tie weights
        else:
            h = self.enc(tokens)
        feat = self.ln(h.mean(dim=1))
        if return_features:
            return self.cls(feat), h
        return self.cls(feat)


class LiteCrossAttnDecoderHead(nn.Module):
    """
    q_theta(y | Z12, Zk): tiny TransformerDecoder with single query token from Z12.
    """
    def __init__(self, comp_dim, num_classes, nhead=2, num_layers=1, ff_mult=2, dropout=0.1, share_layer=True):
        super().__init__()
        ff_dim = ff_mult * comp_dim
        dec_layer = nn.TransformerDecoderLayer(
            d_model=comp_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        if share_layer and num_layers > 1:
            self.dec = nn.TransformerDecoder(dec_layer, num_layers=1)
            self.num_layers = num_layers
            self.shared_layer = dec_layer
        else:
            self.dec = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
            self.num_layers = None
            self.shared_layer = None
        self.q_proj = nn.Linear(comp_dim, comp_dim)  # cheap
        self.ln = nn.LayerNorm(comp_dim)
        self.cls = nn.Linear(comp_dim, num_classes)

    def forward(self, tgt, mem_tokens):  # z12_vec: [B,c], mem_tokens: [B,T,c]
        # tgt = self.q_proj(z12_vec)  # [B,1,c]
        if self.shared_layer is not None:
            h = tgt
            for _ in range(self.num_layers):
                h = self.dec.layers[0](h, mem_tokens)  # tie weights across "layers"
        else:
            h = self.dec(tgt=tgt, memory=mem_tokens)
        h = self.ln(h.squeeze(1))
        return self.cls(h)


# ----------------- Model with lightweight complementarity heads -----------------
class Fusion_Synprom_perf_v2(nn.Module):
    def __init__(self, args, encs):
        super().__init__()
        self.args = args
        self.cls_type = args.cls_type
        self.num_classes = args.num_classes

        d_model  = args.d_model
        fc_inner = args.fc_inner
        dropout  = args.get("dropout", 0.1)

        self.enc_0, self.enc_1, self.enc_2, self.enc_3 = encs

        # existing knobs
        bias_infusion = args.get("bias_infusion", {})
        self.l = bias_infusion.get("l", 0.0)  # your original synergy weight
        self.regby = bias_infusion.get("regby", None)  # your original synergy weight

        # NEW: lightweight complementarity knobs
        self.enable_comp   = args.get("enable_comp", True)
        self.comp_gamma    = args.get("comp_margin", 0.0)     # hinge margin (nats)
        self.comp_sym      = args.get("comp_symmetric", True) # Z2 branch as well
        self.comp_dim      = args.get("comp_dim", min(128, d_model))  # << shrink width here
        self.comp_heads    = args.get("comp_heads", 2)        # smaller heads
        self.comp_layers   = args.get("comp_layers", 1)       # 1 layer
        self.comp_ff_mult  = args.get("comp_ff_mult", 2)      # FFN = 2*dim
        self.comp_share    = args.get("comp_share", True)     # share heads across Z1/Z2
        self.share_layer_tying = args.get("comp_share_layer", True)  # tie within head

        assert self.comp_dim % self.comp_heads == 0, "comp_dim must be divisible by comp_heads"

        # backbone (unchanged)

        self.fusion_head = LiteTransformerBaseline(
            comp_dim=d_model, num_classes=self.num_classes,
            nhead=self.comp_heads, num_layers=self.comp_layers,
            ff_mult=self.comp_ff_mult, dropout=dropout,
            share_layer=self.share_layer_tying
        )

        # Projections to compact space (reduce params a lot)
        # LazyLinear adapts to whatever token dims you bring (e.g., [B,T,F])
        # 1x1 convs on [B, F, S]
        self.proj_z1 = nn.Conv1d(d_model, self.comp_dim, kernel_size=1, bias=False)
        self.proj_z2 = nn.Conv1d(d_model, self.comp_dim, kernel_size=1, bias=False)
        self.proj_z12 = nn.Conv1d(d_model, self.comp_dim, kernel_size=1, bias=False)

        self.ln_z1 = nn.LayerNorm(self.comp_dim)
        self.ln_z2 = nn.LayerNorm(self.comp_dim)
        self.ln_z12 = nn.LayerNorm(self.comp_dim)

        if self.enable_comp:
            # shared heads (halve parameters)
            self.base_head_shared = LiteTransformerBaseline(
                comp_dim=self.comp_dim, num_classes=self.num_classes,
                nhead=self.comp_heads, num_layers=self.comp_layers,
                ff_mult=self.comp_ff_mult, dropout=dropout,
                share_layer=self.share_layer_tying
            )
            self.comp_head_shared = LiteTransformerBaseline(
                comp_dim=self.comp_dim, num_classes=self.num_classes,
                nhead=self.comp_heads, num_layers=self.comp_layers,
                ff_mult=self.comp_ff_mult, dropout=dropout,
                share_layer=self.share_layer_tying
            )

    def _get_features(self, x, **kwargs):
        a = self.enc_0(x, **kwargs)   # modality 1
        v = self.enc_1(x, **kwargs)   # modality 2
        joint_a = self.enc_2(x, **kwargs)
        joint_v = self.enc_3(x, **kwargs)
        return a, v, joint_a, joint_v

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):

        feat = torch.cat((a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]), dim=2).transpose(1, 2).contiguous()

        pred, aggr_feat = self.fusion_head(feat, return_features=True)

        feat_tokens = aggr_feat.mean(dim=1)

        return pred, pred_aa, pred_vv, aggr_feat, feat_tokens

    def forward(self, x, **kwargs):
        a, v, joint_a, joint_v = self._get_features(x, **kwargs)

        pred_a = a["preds"]["combined"]  # [B,C]
        pred_v = v["preds"]["combined"]  # [B,C]

        pred, pred_joint_a, pred_joint_v, z12_tokens, _ = self._forward_main(
            joint_a, joint_v, joint_a["preds"]["combined"], joint_v["preds"]["combined"], **kwargs
        )

        output = {
            "preds": {"combined": pred, "joint_c": pred_joint_a, "joint_g": pred_joint_v, "c": pred_a, "g": pred_v},
            "features": {
                "c": a["features"]["combined"], "g": v["features"]["combined"],
                "joint_c": joint_a["features"]["combined"], "joint_g": joint_v["features"]["combined"],
            }
        }

        if self.training and self.l!=0:
            labels = kwargs["label"].view(-1, 1)

            # --- shape & compact projections ---
            # your tokens are [B,F,S] -> [B,S,F]
            z1_tokens = a["nonaggr_features"]["combined"].contiguous()
            z2_tokens = v["nonaggr_features"]["combined"].contiguous()

            if self.regby == "z12z1_z1":
                losses = self.z12z1_z1_gap(z1_tokens, z2_tokens, z12_tokens, labels)
            elif self.regby == "z12z1mipd":
                losses = self.z12z1mipd(z1_tokens, z2_tokens, z12_tokens, labels)

            output["losses"] = losses

        return output

    def z12z1_z1_gap(self, z1_tokens, z2_tokens, z12_tokens, labels):
        # project to compact comp_dim and norm
        z1_tokens = self.ln_z1(self.proj_z1(z1_tokens).transpose(1, 2))  # [B,T1,c]
        z2_tokens = self.ln_z2(self.proj_z2(z2_tokens).transpose(1, 2))  # [B,T2,c]
        z12_tokens = self.ln_z12(self.proj_z12(z12_tokens.transpose(1, 2)).transpose(1, 2))  # [B,c]

        # --- shared heads ---
        # q_theta(. | Z12, Z1), q_eta(. | Z1)
        logits_theta1 = self.comp_head_shared(torch.cat((z12_tokens, z1_tokens), dim=1))  # [B,C]
        logits_eta1 = self.base_head_shared(z1_tokens)
        logits_eta1 = logits_eta1.detach()  # stop-grad baseline

        logp_theta1 = F.log_softmax(logits_theta1, dim=1).gather(1, labels).squeeze(1)
        logp_eta1 = F.log_softmax(logits_eta1, dim=1).gather(1, labels).squeeze(1)
        # comp1 = F.relu(-logp_theta1 + logp_eta1 + self.comp_gamma).mean()
        comp1 = (-logp_theta1 + logp_eta1).mean()

        losses = {"comp1": self.l * comp1}

        if self.comp_sym:
            # reuse the SAME heads for Z2 branch (parameter sharing)
            logits_theta2 = self.comp_head_shared(torch.cat((z12_tokens, z2_tokens), dim=1))
            logits_eta2 = self.base_head_shared(z2_tokens).detach()

            logp_theta2 = F.log_softmax(logits_theta2, dim=1).gather(1, labels).squeeze(1)
            logp_eta2 = F.log_softmax(logits_eta2, dim=1).gather(1, labels).squeeze(1)
            # comp2 = F.relu(-logp_theta2 + logp_eta2 + self.comp_gamma).mean()
            comp2 = (-logp_theta2 + logp_eta2).mean()
            losses["comp2"] = self.l * comp2
        return losses

    def z12z1mipd(self, z1_tokens, z2_tokens, z12_tokens, labels):
        # ---- project to compact comp_dim and normalize ----
        # inputs expected as [B, F, S] -> proj -> [B, S, c]
        z1_tokens = self.ln_z1(self.proj_z1(z1_tokens).transpose(1, 2))  # [B, T1, c]
        z2_tokens = self.ln_z2(self.proj_z2(z2_tokens).transpose(1, 2))  # [B, T2, c]
        z12_tokens = self.proj_z12(z12_tokens.transpose(1, 2)).transpose(1, 2)  # [B, T12, c] or [B, c]
        z12_tokens = self.ln_z12(z12_tokens)
        if z12_tokens.dim() == 2:  # make sure z12 has time axis
            z12_tokens = z12_tokens.unsqueeze(1)  # [B, 1, c]

        B = z1_tokens.size(0)
        K = getattr(self, "perm_K", self.args.get("perm_K", 4))  # number of permutations

        # ---- positive (clean) log-prob with concatenated tokens ----
        logits_pos1 = self.comp_head_shared(torch.cat((z12_tokens, z1_tokens), dim=1))  # [B, C]
        logp_pos1 = F.log_softmax(logits_pos1, dim=1).gather(1, labels).squeeze(1)

        # ---- build K permutations in a vectorized way (no Python loop over K) ----
        # P: [K, B] each row a permutation of range(B)
        P = torch.argsort(torch.rand(K, B, device=z1_tokens.device), dim=1)  # sampled perms
        # z12_perm: [K, B, T12, c]
        z12_perm = z12_tokens[P]
        # repeat z1 along batch K times to match: [K, B, T1, c]
        z1_rep = z1_tokens.unsqueeze(0).expand(K, -1, -1, -1)
        # flatten first two dims -> [K*B, T*, c]
        z12_perm_flat = z12_perm.reshape(K * B, z12_perm.size(2), z12_perm.size(3))
        z1_rep_flat = z1_rep.reshape(K * B, z1_rep.size(2), z1_rep.size(3))
        labels_flat = labels.expand(K, -1, -1).reshape(K * B, 1)

        # ---- permutation average E_pi[ log q( y | Z12^(pi), Z1 ) ] (vectorized) ----
        logits_perm_flat = self.comp_head_shared(torch.cat((z12_perm_flat, z1_rep_flat), dim=1))  # [K*B, C]
        logp_perm_flat = F.log_softmax(logits_perm_flat, dim=1).gather(1, labels_flat).squeeze(1)
        logp_perm_mean = logp_perm_flat.view(K, B).mean(0)  # [B]

        # loss = E[ log q(y | Z12^pi, Z1) - log q(y | Z12, Z1) ]
        comp1 = (logp_perm_mean - logp_pos1).mean()
        losses = {"comp1": self.l * comp1}

        if self.comp_sym:
            # Positive with Z2
            logits_pos2 = self.comp_head_shared(torch.cat((z12_tokens, z2_tokens), dim=1))
            logp_pos2 = F.log_softmax(logits_pos2, dim=1).gather(1, labels).squeeze(1)

            # Vectorized permutations for Z2 branch
            z2_rep = z2_tokens.unsqueeze(0).expand(K, -1, -1, -1)  # [K,B,T2,c]
            z2_rep_flat = z2_rep.reshape(K * B, z2_rep.size(2), z2_rep.size(3))
            logits_perm2_flat = self.comp_head_shared(torch.cat((z12_perm_flat, z2_rep_flat), dim=1))
            logp_perm2_flat = F.log_softmax(logits_perm2_flat, dim=1).gather(1, labels_flat).squeeze(1)
            logp_perm2_mean = logp_perm2_flat.view(K, B).mean(0)

            comp2 = (logp_perm2_mean - logp_pos2).mean()
            losses["comp2"] = self.l * comp2

        return losses

