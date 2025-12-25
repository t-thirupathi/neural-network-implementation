import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


ParamDict = Dict[str, np.ndarray]


class Optimizer(ABC):
    """
    Base optimizer class.
    Returns param_update to be SUBTRACTED from parameters.
    theta_t+1 = theta_t - param_update
    """

    def __init__(self, params_dict, lr=1e-3):
        self.lr = lr
        self.params_dict = params_dict

    @abstractmethod
    def step(self, param_grads: ParamDict) -> ParamDict:
        """
        Compute parameter updates.
        """
        pass

class GradientDescentOptimizer(Optimizer):
    """
    Vanilla Gradient Descent (full-batch).
    """

    def __init__(self, params_dict, lr=1e-3):
        super().__init__(params_dict, lr)

    def step(self, param_grads: ParamDict) -> ParamDict:
        # theta_t+1 = theta_t - lr * gradient
        param_updates = {}
        for k, grad in param_grads.items():
            param_updates[k] = self.lr * grad
        return param_updates


class MomentumOptimizer(Optimizer):
    MOMENTUM_COEFF = 0.9 # decay rate

    def __init__(self, params_dict, lr=1e-3):
        super().__init__(params_dict, lr)
        self.velocity = {
            k: np.zeros_like(v) for k, v in params_dict.items()
        }

    def step(self, param_grads: ParamDict) -> ParamDict:
        param_updates = {}
        # v_t+1 = decay_rate * v_t + lr * gradient
        for k, grad in param_grads.items():
            self.velocity[k] = (
                self.MOMENTUM_COEFF * self.velocity[k]
                + self.lr * grad
            )
            param_updates[k] = self.velocity[k]
        return param_updates


class AdaGradOptimizer(Optimizer):
    EPS = 1e-8

    def __init__(self, params_dict, lr=1e-3):
        super().__init__(params_dict, lr)
        self.cache = { # accumulated sum of squared gradients
            k: np.zeros_like(v) for k, v in params_dict.items()
        }

    def step(self, param_grads: ParamDict) -> ParamDict:
        param_updates = {}
        # theta_t+1 = theta_t - (lr / (sqrt(sum_of_gradients_squared) + eps)) * gradient
        for k, grad in param_grads.items():
            self.cache[k] += grad ** 2
            adaptive_lr = self.lr / (np.sqrt(self.cache[k]) + self.EPS)
            param_updates[k] = adaptive_lr * grad
        return param_updates


class RMSPropOptimizer(Optimizer):
    DECAY_RATE = 0.99
    EPS = 1e-8

    def __init__(self, params_dict, lr=1e-3):
        super().__init__(params_dict, lr)
        self.cache = {
            k: np.zeros_like(v) for k, v in params_dict.items()
        }

    def step(self, param_grads: ParamDict) -> ParamDict:
        param_updates = {}
        # theta_t+1 = theta_t - (lr / (sqrt(decay_rate * sum_of_gradients_squared + (1 - decay_rate) * gradient^2) + eps)) * gradient
        for k, grad in param_grads.items():
            self.cache[k] = (
                self.DECAY_RATE * self.cache[k]
                + (1 - self.DECAY_RATE) * (grad ** 2)
            )
            adaptive_lr = self.lr / (np.sqrt(self.cache[k]) + self.EPS)
            param_updates[k] = adaptive_lr * grad
        return param_updates


class AdamOptimizer(Optimizer): # combines Momentum and RMSProp
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8

    def __init__(self, params_dict, lr=1e-3):
        super().__init__(params_dict, lr)
        self.m = {
            k: np.zeros_like(v) for k, v in params_dict.items()
        }
        self.v = {
            k: np.zeros_like(v) for k, v in params_dict.items()
        }
        self.t = 0

    def step(self, param_grads: ParamDict) -> ParamDict:
        self.t += 1
        param_updates = {}

        for k, grad in param_grads.items():
            # Update biased first moment estimate
            self.m[k] = (
                self.BETA1 * self.m[k]
                + (1 - self.BETA1) * grad
            )

            # Update biased second moment estimate
            self.v[k] = (
                self.BETA2 * self.v[k]
                + (1 - self.BETA2) * (grad ** 2)
            )

            # Bias correction
            m_hat = self.m[k] / (1 - self.BETA1 ** self.t)
            v_hat = self.v[k] / (1 - self.BETA2 ** self.t)

            # Parameter update
            param_updates[k] = (
                self.lr * m_hat / (np.sqrt(v_hat) + self.EPS)
            )

        return param_updates
