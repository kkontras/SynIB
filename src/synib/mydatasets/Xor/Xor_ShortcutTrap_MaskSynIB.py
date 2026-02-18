import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


# -------------------------
# 1. Configuration
# -------------------------
@dataclass
class Config:
    seed: int = 42
    dim0: int = 64
    dim1: int = 64
    signal_dims: int = 4
    spur_dims: int = 6
    alpha_train: float = 0.8
    n_train: int = 2000
    n_test: int = 1000
    batch_size: int = 128

    pred_hidden: int = 256
    pred_epochs: int = 20
    pred_lr: float = 5e-4

    lambda_uni: float = 1.0
    lambda_kl: float = 10.0  # Increased to enforce counterfactuals
    mask_inner_steps: int = 100
    mask_lr: float = 0.1


class MultimodalDataset(Dataset):
    def __init__(self, n, dim0, dim1, signal_dims, spur_dims, alpha, seed=0):
        rng = np.random.RandomState(seed)
        self.y = torch.from_numpy(rng.choice([0.0, 1.0], size=(n, 1)).astype(np.float32))
        y_sign = (self.y * 2 - 1)

        self.x0 = torch.randn(n, dim0) * 0.5
        self.x0[:, :signal_dims] += 1.5 * torch.randn(n, signal_dims) * y_sign
        if spur_dims > 0:
            mask = (torch.from_numpy(rng.rand(n, 1)) < alpha).float()
            spur_y = mask * y_sign + (1 - mask) * (torch.randint(0, 2, (n, 1)) * 2 - 1)
            self.x0[:, signal_dims:signal_dims + spur_dims] += 1.5 * torch.randn(n, spur_dims) * spur_y

        self.x1 = torch.randn(n, dim1) * 0.5
        self.x1[:, :signal_dims] += 1.5 * torch.randn(n, signal_dims) * y_sign

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return {"x0": self.x0[idx], "x1": self.x1[idx], "y": self.y[idx]}


# -------------------------
# 3. Model & Trainer
# -------------------------
class SynIBModel(nn.Module):
    def __init__(self, dim0, dim1, hidden, feat_drop_prob=0.2, mod_drop_prob=0.3):
        super().__init__()
        # Feature-level dropout layers
        self.drop0 = nn.Dropout(p=feat_drop_prob)
        self.drop1 = nn.Dropout(p=feat_drop_prob)
        self.mod_drop_prob = mod_drop_prob

        # Heads
        self.f = nn.Sequential(
            nn.Linear(dim0 + dim1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.u0 = nn.Sequential(nn.Linear(dim0, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.u1 = nn.Sequential(nn.Linear(dim1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward_logits(self, x0, x1, training=True):
        # 1. Feature-level dropout (Applied to raw inputs)
        x0_f = self.drop0(x0) if training else x0
        x1_f = self.drop1(x1) if training else x1

        # 2. Modality-level dropout (Only for the Fusion Head)
        # We create copies for fusion so we don't break the unimodal probes
        x0_for_f, x1_for_f = x0_f, x1_f

        # if training:
        #     r = torch.rand(1).item()
        #     if r < self.mod_drop_prob:
        #         # Drop Modality 0 (Forces f to look at Modality 1)
        #         x0_for_f = torch.zeros_like(x0_f)
        #     elif r < 2 * self.mod_drop_prob:
        #         # Drop Modality 1 (Forces f to look at Modality 0)
        #         x1_for_f = torch.zeros_like(x1_f)

        f_logits = self.f(torch.cat([x0_for_f, x1_for_f], dim=-1))

        # Unimodal probes always see their respective (feature-dropped) data
        u0_logits = self.u0(x0_f)
        u1_logits = self.u1(x1_f)

        return f_logits, u0_logits, u1_logits


class SynIBTrainer:
    def __init__(self, model, cfg: Config):
        self.model = model
        self.cfg = cfg
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.pred_lr)

    def kl_bern_to_uniform(self, logits):
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        return (p * torch.log(p * 2) + (1 - p) * torch.log((1 - p) * 2)).mean()

    def get_adversarial_masks(self, x0, x1):
        # Initialize at 1.0 (keep everything)
        m0 = nn.Parameter(torch.ones_like(x0) * 3.0, requires_grad=True)
        m1 = nn.Parameter(torch.ones_like(x1) * 3.0, requires_grad=True)
        m_opt = optim.Adam([m0, m1], lr=self.cfg.mask_lr)

        with torch.no_grad():
            _, u0_clean, u1_clean = self.model.forward_logits(x0, x1)
            u0_target, u1_target = torch.sigmoid(u0_clean), torch.sigmoid(u1_clean)

        self.model.eval()
        for _ in range(self.cfg.mask_inner_steps):
            m_opt.zero_grad()
            # m0, m1 = torch.sigmoid(m0_logits / 0.1), torch.sigmoid(m1_logits / 0.1)
            # m0, m1 = torch.sigmoid(m0_logits / 0.1), torch.sigmoid(m1_logits / 0.1)
            # m0, m1 = torch.sigmoid(m0), torch.sigmoid(m1)

            f_m0, u0_m0, _ = self.model.forward_logits(torch.sigmoid(m0) * x0, x1)
            f_m1, _, u1_m1 = self.model.forward_logits(x0, torch.sigmoid(m1) * x1)

            l_break = self.kl_bern_to_uniform(f_m0) + self.kl_bern_to_uniform(f_m1)
            l_keep = F.mse_loss(torch.sigmoid(u0_m0), u0_target) + F.mse_loss(torch.sigmoid(u1_m1), u1_target)
            l_stay = F.mse_loss(m0, torch.ones_like(m0)) + F.mse_loss(m1, torch.ones_like(m1))

            # Maximize l_break (destroy fusion) while minimizing l_keep (preserve unimodal)
            (5000 * l_break + 1 * l_keep + 0.01 * l_stay).backward()
            m_opt.step()

        return (m0 > 0.5).float().detach(), (m1 > 0.5).float().detach()
        # return torch.sigmoid(m0_logits).detach(), torch.sigmoid(m1_logits).detach()

    def calculate_acc(self, loader, device):
        self.model.eval()
        f_hits, u0_hits, total = 0, 0, 0
        with torch.no_grad():
            for b in loader:
                x0, x1, y = b['x0'].to(device), b['x1'].to(device), b['y'].to(device)
                f, u0, _ = self.model.forward_logits(x0, x1)
                f_hits += ((torch.sigmoid(f) > 0.5).float() == y).sum().item()
                u0_hits += ((torch.sigmoid(u0) > 0.5).float() == y).sum().item()
                total += y.size(0)
        return f_hits / total, u0_hits / total

    def train_epoch(self, loader, device):
        self.model.train()
        cov_stats = []

        for batch in loader:
            x0, x1, y = batch['x0'].to(device), batch['x1'].to(device), batch['y'].to(device)
            m0, m1 = self.get_adversarial_masks(x0, x1)

            # Log coverage for modality 0
            comp_cov = m0[:, :self.cfg.signal_dims].mean().item()
            spur_cov = m0[:, self.cfg.signal_dims:self.cfg.signal_dims + self.cfg.spur_dims].mean().item()
            noise_cov = m0[:, self.cfg.signal_dims + self.cfg.spur_dims:].mean().item()
            cov_stats.append([comp_cov, spur_cov, noise_cov])
            print(f"Coverage m0 - Comp: {comp_cov:.3f}, Spur: {spur_cov:.3f}, Noise: {noise_cov:.3f}")

            self.model.train()
            self.optimizer.zero_grad()
            f_logits, u0_logits, u1_logits = self.model.forward_logits(x0, x1)

            l_base = F.binary_cross_entropy_with_logits(f_logits, y) + \
                     self.cfg.lambda_uni * (F.binary_cross_entropy_with_logits(u0_logits, y) +
                                            F.binary_cross_entropy_with_logits(u1_logits, y))

            f_tilde0, _, _ = self.model.forward_logits(m0 * x0, x1)
            f_tilde1, _, _ = self.model.forward_logits(x0, m1 * x1)
            l_cf = self.kl_bern_to_uniform(f_tilde0) + self.kl_bern_to_uniform(f_tilde1)

            (l_base + self.cfg.lambda_kl * l_cf).backward()
            self.optimizer.step()

        return np.mean(cov_stats, axis=0)


# -------------------------
# 4. Main
# -------------------------
def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    train_loader = DataLoader(
        MultimodalDataset(cfg.n_train, cfg.dim0, cfg.dim1, cfg.signal_dims, cfg.spur_dims, cfg.alpha_train),
        batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(
        MultimodalDataset(cfg.n_test, cfg.dim0, cfg.dim1, cfg.signal_dims, cfg.spur_dims, alpha=0.0),
        batch_size=cfg.batch_size, shuffle=False)

    model = SynIBModel(cfg.dim0, cfg.dim1, cfg.pred_hidden).to(device)
    trainer = SynIBTrainer(model, cfg)

    print(f"{'Epoch':<4} | {'Tr_Fus':<6} | {'Tr_Uni':<6} | {'Te_Fus':<6} | {'Te_Uni':<6} || {'M0_Comp':<7} | {'M0_Spur':<7} | {'M0_Noise':<7}")
    print("-" * 65)

    for epoch in range(cfg.pred_epochs):
        avg_cov = trainer.train_epoch(train_loader, device)
        tr_f, tr_u = trainer.calculate_acc(train_loader, device)
        te_f, te_u = trainer.calculate_acc(test_loader, device)

        print(
            f"{epoch + 1:02d}   | {tr_f:.3f} | {tr_u:.3f} | {te_f:.3f} | {te_u:.3f} || {avg_cov[0]:.3f}   | {avg_cov[1]:.3f}   | {avg_cov[2]:.3f}")


if __name__ == "__main__":
    main()