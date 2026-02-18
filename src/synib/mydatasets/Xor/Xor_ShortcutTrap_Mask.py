import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass



# -------------------------
# 1. Config & Hyperparameters
# -------------------------

@dataclass
class MaskWeights:
    kl_break: float = 2.0      # Cranked up to force complementary signal to 0
    keep_u0: float = 5.0       # Cranked up to protect spurious/unimodal info
    sparsity: float = 0.00     # Lowered: let kl_break do the heavy lifting
    lr: float = 0.2            # Faster convergence
    steps: int = 200           # More steps for "hard" convergence

@dataclass
class Config:
    seed: int = 42
    dim0: int = 64
    dim1: int = 64
    signal_dims: int = 4
    spur_dims: int = 6
    alpha_train: float = 0.8
    n_train: int = 2000
    batch_size: int = 256

    # Predictor params
    pred_hidden: int = 256
    pred_epochs: int = 2
    pred_lr: float = 1e-3


# -------------------------
# 2. Synthetic Dataset
# -------------------------

class MultimodalDataset(Dataset):
    def __init__(self, n, dim0, dim1, signal_dims, spur_dims, alpha, seed=0):
        rng = np.random.RandomState(seed)
        self.y = torch.from_numpy(rng.choice([0.0, 1.0], size=(n, 1)).astype(np.float32))

        # Base noise
        self.x0 = torch.randn(n, dim0)
        self.x1 = torch.randn(n, dim1)

        # Complementary Signal (Shared)
        sig = torch.randn(n, signal_dims)
        y_sign = (self.y * 2 - 1)
        self.x0[:, :signal_dims] += 2.0 * sig * y_sign
        self.x1[:, :signal_dims] += 2.0 * sig * y_sign

        # Spurious correlation (Modality 0 only)
        if spur_dims > 0:
            spur = torch.randn(n, spur_dims)
            # alpha chance of being correct label
            mask = (torch.rand(n, 1) < alpha).float()
            spur_y = mask * y_sign + (1 - mask) * (torch.randint(0, 2, (n, 1)) * 2 - 1)
            self.x0[:, signal_dims:signal_dims + spur_dims] += 2.0 * spur * spur_y

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return {"x0": self.x0[idx], "x1": self.x1[idx], "y": self.y[idx]}


class MultimodalPredictor(nn.Module):
    def __init__(self, dim0, dim1, hidden):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim0 + dim1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.u0 = nn.Sequential(nn.Linear(dim0 + dim1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward_logits(self, x0, x1):
        x = torch.cat([x0, x1], dim=-1)
        return self.f(x), self.u0(x)


class DirectMaskOptimizer:
    def __init__(self, predictor: nn.Module, weights: MaskWeights):
        self.predictor = predictor
        self.weights = weights

    def optimize_batch(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Start at 0.5 (Neutral) so the model can easily move to 0 or 1
        mask_params = nn.Parameter(torch.zeros_like(x0))
        opt = optim.Adam([mask_params], lr=0.2)

        with torch.no_grad():
            _, clean_u0_logits = self.predictor.forward_logits(x0, x1)
            target_u0_probs = torch.sigmoid(clean_u0_logits)

        for step in range(200):
            opt.zero_grad()

            # Sharpness increases over time to force a decision
            temp = max(1.0 - (step / 200), 0.1)
            m = torch.sigmoid(mask_params / temp)

            x0_masked = m * x0
            f_logits, u0_logits = self.predictor.forward_logits(x0_masked, x1)

            # 1. THE BREAK (Must be violent)
            # We use a very aggressive 'Max-Entropy' loss
            # Target fusion probability: 0.5 (Maximum uncertainty)
            l_break = F.binary_cross_entropy_with_logits(
                f_logits, torch.full_like(f_logits, 0.5)
            )

            # 2. THE PROTECTOR (Protect Spurious)
            l_keep_u0 = F.mse_loss(torch.sigmoid(u0_logits), target_u0_probs)

            # 3. THE BUDGET (The "Anti-1.0" Force)
            # We force the average mask value to be around 0.9.
            # This forces the model to choose ~10% of features to kill.
            # If everything is 1.0, this loss is high.
            target_sparsity = 1.0
            l_budget = F.mse_loss(m.mean(), torch.tensor(target_sparsity).to(m.device))

            # 4. BINARY ENFORCER (Prevents 0.5 stagnation)
            l_binary = (m * (1 - m)).mean()

            # Balancing the Tug-of-War
            # Break is the strongest, Budget forces 'something' to disappear,
            # Keep-U0 ensures that 'something' isn't a spurious feature.
            total_loss = (100.0 * l_break) + ( 100 * l_keep_u0) + ( 1 * l_budget)

            total_loss.backward()
            opt.step()

        return (torch.sigmoid(mask_params / 0.1) > 0.5).float().detach()

# Logic check for the main script:
def analyze_mask(final_mask, cfg):
    m_np = final_mask.cpu().numpy()
    comp = m_np[:, :cfg.signal_dims].mean()
    spur = m_np[:, cfg.signal_dims:cfg.signal_dims + cfg.spur_dims].mean()
    noise = m_np[:, cfg.signal_dims + cfg.spur_dims:].mean()

    print("\n--- Mask Results ---")
    print(f"Complementary (Goal 0.0): {comp:.3f}")
    print(f"Spurious      (Goal 1.0): {spur:.3f}")
    print(f"Noise         (Goal 1.0): {noise:.3f}")# 5. Main Execution Script
# -------------------------

def calculate_accuracy(model, loader, device):
    model.eval()
    f_correct = 0
    u0_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x0, x1, y = batch['x0'].to(device), batch['x1'].to(device), batch['y'].to(device)
            f_logits, u0_logits = model.forward_logits(x0, x1)

            # Convert logits to binary predictions (0 or 1)
            f_pred = (torch.sigmoid(f_logits) > 0.5).float()
            u0_pred = (torch.sigmoid(u0_logits) > 0.5).float()

            f_correct += (f_pred == y).sum().item()
            u0_correct += (u0_pred == y).sum().item()
            total += y.size(0)

    return f_correct / total, u0_correct / total




def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    # 1. Setup Data
    train_ds = MultimodalDataset(cfg.n_train, cfg.dim0, cfg.dim1, cfg.signal_dims, cfg.spur_dims, cfg.alpha_train)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    # 2. Train Predictor
    print(f"--- Training Predictor on {device} ---")
    model = MultimodalPredictor(cfg.dim0, cfg.dim1, cfg.pred_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.pred_lr)

    model.train()
    for epoch in range(cfg.pred_epochs):
        for batch in loader:
            x0, x1, y = batch['x0'].to(device), batch['x1'].to(device), batch['y'].to(device)
            f_logits, u0_logits = model.forward_logits(x0, x1)
            loss = F.binary_cross_entropy_with_logits(f_logits, y) + \
                   F.binary_cross_entropy_with_logits(u0_logits, y)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
        print(f"Epoch {epoch + 1}/{cfg.pred_epochs} Loss: {loss.item():.4f}")

    test_ds = MultimodalDataset(
        n=1000,
        dim0=cfg.dim0,
        dim1=cfg.dim1,
        signal_dims=cfg.signal_dims,
        spur_dims=cfg.spur_dims,
        alpha=0.0,  # Test set usually has no spurious correlation
        seed=99
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    train_f_acc, train_u0_acc = calculate_accuracy(model, loader, device)
    test_f_acc, test_u0_acc = calculate_accuracy(model, test_loader, device)

    print(f"\n{'=' * 30}")
    print(f"PREDICTOR ACCURACY")
    print(f"{'-' * 30}")
    print(f"Train Fusion Acc:  {train_f_acc:.4f}")
    print(f"Train Unimodal Acc: {train_u0_acc:.4f}")
    print(f"{'-' * 30}")
    print(f"Test Fusion Acc:   {test_f_acc:.4f}")
    print(f"Test Unimodal Acc:  {test_u0_acc:.4f}")
    print(f"{'=' * 30}\n")

    # 3. Direct Mask Optimization (The Goal)
    print("\n--- Running Direct Mask Optimization ---")
    model.eval()
    mask_weights = MaskWeights()  # Tune this to control mask strength
    mask_opt = DirectMaskOptimizer(model, mask_weights)

    # Take one batch to visualize result
    sample_batch = next(iter(loader))
    x0, x1 = sample_batch['x0'].to(device), sample_batch['x1'].to(device)

    final_mask = mask_opt.optimize_batch(x0, x1)

    # Analyze Coverage (1.0 = kept, 0.0 = masked)
    def analyze(m):
        m_np = m.cpu().numpy()
        comp = m_np[:, :cfg.signal_dims].mean()
        spur = m_np[:, cfg.signal_dims:cfg.signal_dims + cfg.spur_dims].mean()
        rest = m_np[:, cfg.signal_dims + cfg.spur_dims:].mean()
        return comp, spur, rest

    comp, spur, rest = analyze(final_mask)

    print("\nResults (Average Keep-Probability):")
    print(f"Complementary Signal Dims: {comp:.3f} (Lower means it's successfully masked)")
    print(f"Spurious Dims:             {spur:.3f} (Higher means it's preserved)")
    print(f"Noise Dims:                {rest:.3f}")

    print("\nInterpretation:")
    if comp < spur:
        print(">> Success: The optimizer prioritized masking the complementary signal!")
    else:
        print(">> Increase 'sparsity' or 'kl_break' weights to force more masking.")


if __name__ == "__main__":
    main()