from typing import Mapping
import numpy as np


class Optimizer:
    """
    Base optimizer class.
    Returns parameter updates to be SUBTRACTED from parameters.
    """

    def __init__(self, params_dict: Mapping[str, np.array], lr=1e-3):
        self.lr = lr
        self.params_dict = params_dict

    def step(self, param_grads: Mapping[str, np.array]) -> Mapping[str, np.array]:
        param_updates = {}
        for param_name, grad in param_grads.items():
            param_updates[param_name] = self.lr * grad
        return param_updates


# ---------------- Momentum ---------------- #

class MomentumOptimizer(Optimizer):
    MOMENTUM_COEFF = 0.9

    def __init__(self, params_dict: Mapping[str, np.array], lr=1e-3):
        super().__init__(params_dict, lr)
        self.velocity = {
            k: np.zeros_like(v) for k, v in params_dict.items()
        }

    def step(self, param_grads: Mapping[str, np.array]) -> Mapping[str, np.array]:
        param_updates = {}
        for k, grad in param_grads.items():
            self.velocity[k] = (
                self.MOMENTUM_COEFF * self.velocity[k]
                + self.lr * grad
            )
            param_updates[k] = self.velocity[k]
        return param_updates


# ---------------- RMSProp ---------------- #

class RMSPropOptimizer(Optimizer):
    DECAY_RATE = 0.99
    EPS = 1e-8

    def __init__(self, params_dict: Mapping[str, np.array], lr=1e-3):
        super().__init__(params_dict, lr)
        self.cache = {
            k: np.zeros_like(v) for k, v in params_dict.items()
        }

    def step(self, param_grads: Mapping[str, np.array]) -> Mapping[str, np.array]:
        param_updates = {}
        for k, grad in param_grads.items():
            self.cache[k] = (
                self.DECAY_RATE * self.cache[k]
                + (1 - self.DECAY_RATE) * (grad ** 2)
            )
            adaptive_lr = self.lr / (np.sqrt(self.cache[k]) + self.EPS)
            param_updates[k] = adaptive_lr * grad
        return param_updates
