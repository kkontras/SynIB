"""Real linear-warmup scheduler.

`WarmupScheduler` (the existing class) holds LR constant at base_lr for the
first `n_warmup_steps`, which is not a warmup. This class linearly ramps from
0 -> base_lr over `n_warmup_steps`, then holds at base_lr.

Used to escape the LoRA zero-init saddle for plain LoRA at r >= 16 on TQA,
where the model can otherwise stay at uniform-output (ce ~= ln K) for the
entire training budget.
"""

import torch.nn as nn
import wandb


class LinearWarmupConstant(nn.Module):
    def __init__(self, optimizer, base_lr, n_warmup_steps):
        super().__init__()
        self._optimizer = optimizer
        self.base_lr = float(base_lr)
        self.n_warmup_steps = max(1, int(n_warmup_steps))
        self.n_steps = 0
        self.lr_history = []

    def step(self, step=None, loss=None):
        self._update_learning_rate()

    def _update_learning_rate(self):
        self.n_steps += 1
        if self.n_steps <= self.n_warmup_steps:
            lr = self.base_lr * (self.n_steps / self.n_warmup_steps)
        else:
            lr = self.base_lr
        for pg in self._optimizer.param_groups:
            pg['lr'] = lr
        self.lr_history.append(lr)
        try:
            wandb.log({"lr": lr})
        except Exception:
            pass

    def state_dict(self):
        return {"n_steps": self.n_steps, "lr_history": self.lr_history}

    def load_state_dict(self, state):
        self.n_steps = int(state.get("n_steps", 0))
        self.lr_history = list(state.get("lr_history", []))
