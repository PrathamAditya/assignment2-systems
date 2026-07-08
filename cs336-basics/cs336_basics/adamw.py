from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas, eps, weight_decay, lr=1e-3):
        beta_1, beta_2 = betas
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid learning rate: {eps}")
        if not (0 <= beta_1 < 1):
            raise ValueError(f"Invalid learning rate: {beta_1}")
        if not (0 <= beta_2 < 1):
            raise ValueError(f"Invalid learning rate: {beta_2}")
        defaults = {"lr": lr, "beta_1": beta_1, "beta_2": beta_2, "epsilon": eps, "lamda": weight_decay}
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # θ ← θ − αλθ
            lr = group["lr"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                p.data = p.data - group["lr"]*group["lamda"]*p.data
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                    state["t"] = 0
                    state["v"] = torch.zeros_like(p.data)

                state["t"] = state["t"] + 1
                alpha_t= group["lr"]*(math.sqrt(1-group["beta_2"]**state["t"])/(1 - group["beta_1"]**state["t"]))
                state["m"] = group["beta_1"]*state["m"] + (1 - group["beta_1"]) * p.grad.data
                state["v"] = group["beta_2"]*state["v"] + (1 - group["beta_2"]) * (p.grad.data**2)
                p.data = p.data - alpha_t*(state["m"]/(torch.sqrt(state["v"]) + group["epsilon"]))
        return loss
