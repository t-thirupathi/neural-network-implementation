from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """
    Base activation class.
    """

    @staticmethod
    @abstractmethod
    def forward(x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """
        Derivative w.r.t. pre-activation input x
        """
        pass

class Sigmoid(Activation):

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        s = Sigmoid.forward(x)
        return s * (1 - s)

class Tanh(Activation):

    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x):
        return 1 - np.tanh(x) ** 2

class ReLU(Activation):

    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(x):
        return (x > 0).astype(float)

class Swish(Activation):

    @staticmethod
    def forward(x):
        s = 1 / (1 + np.exp(-x))
        return x * s

    @staticmethod
    def backward(x):
        s = 1 / (1 + np.exp(-x))
        return s + x * s * (1 - s)

class LeakyReLU(Activation):

    @staticmethod
    def forward(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def backward(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx