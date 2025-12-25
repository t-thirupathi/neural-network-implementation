from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """
    Base activation class.
    All activations must implement forward and backward.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass (apply activation function)
        """
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative w.r.t. pre-activation input x
        """
        pass


class Sigmoid(Activation):
    def forward(self, x):
        # Numerical stability
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x) ** 2


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        # Gradient at x == 0 is defined as 0 (standard practice)
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        return dx


class Swish(Activation):
    def forward(self, x):
        # x * sigmoid(x)
        sig = 1 / (1 + np.exp(-x))
        return x * sig

    def backward(self, x):
        sig = 1 / (1 + np.exp(-x))
        return sig + x * sig * (1 - sig)

class Softmax(Activation):
    def forward(self, x):
        # Numerical stability
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x):
        # Note: Softmax derivative is usually computed with cross-entropy loss
        s = self.forward(x)
        return s * (1 - s)  # This is a simplification and not the full Jacobian
