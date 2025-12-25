#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

from activation import *
from optimizer import *

# ---------------- Termination ---------------- #
class TerminationCriteria:
    def __init__(self, max_iter, min_delta=1e-6, patience=50):
        self.max_iter = max_iter
        self.min_delta = min_delta
        self.patience = patience # Number of iterations to wait without improvement
        self.patience_counter = 0
        self.best_loss = np.inf

    def is_over(self, iter: int, current_loss: float) -> bool:
        if iter >= self.max_iter:
            return True

        improvement = self.best_loss - current_loss
        if improvement >= self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience


# ---------------- Model ---------------- #
class NeuralNetSolver:
    INPUT_DIM = 2
    HIDDEN_DIM = 3
    OUTPUT_DIM = 1

    MAX_ITER = 10
    INIT_RANGE = 0.1
    NUM_EVALS = 10

    def __init__(self):
        # Randomly initialize weights
        self.params_dict = {
            'weights_1': np.random.normal(0, self.INIT_RANGE,
                                          (self.INPUT_DIM, self.HIDDEN_DIM)),
            'bias_1': np.zeros((1, self.HIDDEN_DIM)),
            'weights_2': np.random.normal(0, self.INIT_RANGE,
                                          (self.HIDDEN_DIM, self.OUTPUT_DIM)),
            'bias_2': np.zeros((1, self.OUTPUT_DIM))
        }
        print("After random initialization - self.params_dict:\n", self.params_dict)
        self.activation = Sigmoid()
        self.optimizer = AdamOptimizer(self.params_dict, lr=1e-3)
        self.termination_criteria = TerminationCriteria(self.MAX_ITER)

    # -------- Forward -------- #
    def forward(self, x):
        # Forward pass for hidden layer
        z1 = x @ self.params_dict['weights_1'] + self.params_dict['bias_1'] # @ is matrix multiplication
        a1 = self.activation.forward(z1)

        # Linear output (regression)
        y_hat = a1 @ self.params_dict['weights_2'] + self.params_dict['bias_2']
        return y_hat

    # -------- Loss -------- #
    @staticmethod
    def get_loss(y, y_hat):
        return np.mean((y - y_hat) ** 2)

    # -------- Backprop: get gradients -------- #
    def get_loss_grad(self, X, y):
        B = X.shape[0]

        # Forward cache
        z1 = X @ self.params_dict['weights_1'] + self.params_dict['bias_1']
        a1 = self.activation.forward(z1)
        y_hat = a1 @ self.params_dict['weights_2'] + self.params_dict['bias_2']

        # MSE Loss: L = (1/B) * sum((y - y_hat)^2)
        # dL/dz2 (or d_z2 in short) = dL/dy_hat (since output layer is linear)
        d_z2 = 2 * (y_hat - y)

        # Layer 2 grads
        # z2 = a1 * W2 + b2  =>  dz2/dw2 = a1, dz2/db2 = 1
        # by chain rule:
        # dL/dw2 = dL/dz2 * dz2/dw2 = a1.T @ dL/dz2
        # dL/db2 = dL/dz2 * dz2/db2 = sum over batch of dL/dz2  
        w2_grad = a1.T @ d_z2 / B
        b2_grad = np.sum(d_z2, axis=0, keepdims=True) / B

        # Backprop to layer 1
        # z1 = X * W1 + b1  =>  dz1/dw1 = X, dz1/db1 = 1
        # by chain rule:
        # dL/da1 = dL/dz2 * dz2/da1 = d_z2 @ W2.T
        # dL/dz1 = dL/da1 * da1/dz1 = dL/da1 * activation.backward(z1)
        d_a1 = d_z2 @ self.params_dict['weights_2'].T
        d_z1 = d_a1 * self.activation.backward(z1)

        # Layer 1 grads
        # dL/dw1 = dL/dz1 * dz1/dw1 = X.T @ dL/dz1
        # dL/db1 = dL/dz1 * dz1/db1 = sum over batch of dL/dz1
        w1_grad = X.T @ d_z1 / B
        b1_grad = np.sum(d_z1, axis=0, keepdims=True) / B

        return {
            'weights_1': w1_grad,
            'bias_1': b1_grad,
            'weights_2': w2_grad,
            'bias_2': b2_grad
        }

    # -------- Train -------- #
    def fit(self, X_train, y_train, X_test, y_test):
        train_losses, test_losses = [], []
        steps_for_eval = self.MAX_ITER // self.NUM_EVALS

        train_loss = np.inf
        iter = 0

        while not self.termination_criteria.is_over(iter, train_loss):
            y_hat = self.forward(X_train) # use full batch
            train_loss = self.get_loss(y_train, y_hat)

            grads = self.get_loss_grad(X_train, y_train)
            updates = self.optimizer.step(grads)

            for k in self.params_dict:
                self.params_dict[k] -= updates[k]

            if iter % steps_for_eval == 0:
                test_pred = self.forward(X_test)
                test_loss = self.get_loss(y_test, test_pred)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                print("self.params_dict:\n", self.params_dict)
                print(f"Iter {iter:6d} | Train {train_loss:.6f} | Test {test_loss:.6f}")

            iter += 1

        return train_losses, test_losses

# ---------------- Data ---------------- #
x1 = np.expand_dims(np.linspace(0, 100, 11), -1)
x2 = np.expand_dims(np.linspace(-50, 50, 11), -1)
X = np.concatenate([x1, x2], axis=1)
y = np.expand_dims(5 * X[:, 0] + 3 * X[:, 0] ** 2 + 50, -1)

# Normalize input
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()

X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------- Train ---------------- #
nn = NeuralNetSolver()
train_losses, test_losses = nn.fit(X_train, y_train, X_test, y_test)

# ---------------- Plots ---------------- #
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.legend()
plt.xlabel(f"Eval step (1 step = {nn.NUM_EVALS} iters)")
plt.ylabel("MSE Loss")
plt.title("Training Curve")

plt.figure()
pred = nn.forward(X)
plt.scatter(y, pred)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("True vs Predicted")
plt.show()
