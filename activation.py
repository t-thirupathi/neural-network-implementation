from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """
    Base activation class.
    """

    @staticmethod
    @abstractmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Forward pass (apply activation function)
        """
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
        sigmoid = 1 / (1 + np.exp(-x))
        return x * sigmoid

    @staticmethod
    def backward(x):
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid + x * sigmoid * (1 - sigmoid)

class LeakyReLU(Activation):

    @staticmethod
    def forward(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def backward(x, alpha=0.01):
        # Derivative is 1 for x > 0, alpha for x <= 0
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx