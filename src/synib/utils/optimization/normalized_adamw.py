import torch
from torch.optim import Optimizer


class NormalizedAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that re-evaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Apply Weight Decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update Adam moving averages
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 1. Compute the Adam direction
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                update = (exp_avg / bias_correction1) / denom

                # 2. Normalize the update to a fixed "Step Length"
                # This aligns the weight updates with your u_star oracle logic
                update_norm = update.norm().clamp_min(group['eps'])

                # 3. Apply the step
                p.add_(update / update_norm, alpha=-group['lr'])

        return loss