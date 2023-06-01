#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from typing import Tuple, List, Collection, Mapping

from optimizer import Optimizer, MomentumOptimizer, RMSPropOptimizer


class TerminationCriteria:
    def __init__(self, max_iter, min_delta=1e-4, patience=10):
        self.max_iter = max_iter
        self.min_delta = min_delta
        self.patience = patience
        self.patience_counter = 0
        self.best_loss = np.inf

    def is_over(self, iter: int, current_loss: float) -> bool:
        """
        : iter
            The iteration number
        :return:
            True if the optimization should terminate, False otherwise.
        """

        if iter > self.max_iter:
            return True

        # Check for minimum improvement in loss
        loss_improvement = self.best_loss - current_loss
        if loss_improvement >= self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            return True

        return False


"""# Model"""

class NeuralNetSolver:
    HIDDEN_DIM = 32
    INPUT_DIM = 1
    OUTPUT_DIM = 1    
    
    MAX_ITER = 100000
    MIN_TRAIN_LOSS = 1e-6
    INIT_RANGE = 1
    NUM_EVALS = 100

    def __init__(self):
        """
        C'tor
        :param input_dim:
            The dimension of the input vector
        """
        self.params_dict = {
            'weights_1': np.random.normal(0, self.INIT_RANGE, (self.INPUT_DIM, self.HIDDEN_DIM)),
            'bias_1': np.zeros(shape=(1, self.HIDDEN_DIM)),
            'weights_2': np.random.normal(0, self.INIT_RANGE, (self.HIDDEN_DIM, self.OUTPUT_DIM)),
            'bias_2': np.zeros(shape=(1, self.OUTPUT_DIM))}

        self.optimizer = RMSPropOptimizer(self.params_dict)
        self.termination_criteria = TerminationCriteria(self.MAX_ITER)

    @staticmethod
    def act_fn(x: np.array) -> np.array:
        """
            The activation function.
        :param x:
            An input array
        :return:
            The result of activation function applied to the x element-wise.
        """
        return x / (1 + np.exp(-x))

    def forward(self, x: np.array) -> np.array:
        """
        Makes a prediction with the model.
        :param x:
            A batch of input feature vectors. The shape is [B, D_IN], where B is the batch dimension
            and D_IN is the dimension of the feature vector.
        :return:
            A batch of predictions. The shape is [B, D_OUT], where B is the batch dimension
            and D_OUT is the dimension of the prediction vector.
        """
        assert x.shape[1] == self.params_dict['weights_1'].shape[0]

        layer_1 = np.dot(x, self.params_dict['weights_1']) + self.params_dict['bias_1']
        a1 = self.act_fn(layer_1)
        
        layer_2 = np.dot(a1, self.params_dict['weights_2']) + self.params_dict['bias_2']
        a2 = self.act_fn(layer_2)
        # a2 = layer_2

        result = a2
        # result = np.zeros((x.shape[0], self.OUTPUT_DIM))

        assert result.ndim == 2
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == self.OUTPUT_DIM

        return result

    def get_loss_grad(self, X: np.array, y: np.array) -> Mapping[str, np.array]:
        """
            Calculates the gradient of the loss function.
        :param X:
            A batch of input feature vectors. The shape is [B, D_IN], where B is the batch dimension
            and D_IN is the dimension of the feature vector.
        :param y:
            A batch of outputs. The shape is [B, D_OUT], where B is the batch dimension
            and D_OUT is the dimension of the output vector.
        :return:
            A
        """
        layer_1_out = X @ self.params_dict['weights_1'] + self.params_dict['bias_1']
        layer_2_out = self.forward(X)

        weights1_grad, bias1_grad = self.get_layer_1_grads(X, y, layer_1_out, layer_2_out)
        weights2_grad, bias2_grad = self.get_layer_2_grads(X, y, layer_1_out, layer_2_out)

        return {'weights_1': weights1_grad, 'bias_1': bias1_grad,
                'weights_2': weights2_grad, 'bias_2': bias2_grad}

    def get_layer_1_grads(self, X: np.array, y: np.array,
                          layer_1_out: np.array, layer_2_out: np.array) -> Tuple[
        np.array, np.array]:
        """
            Calculates the gradients for the first layer of the neural network.
        :param X:
            A batch of input feature vectors. The shape is [B, D_IN], where B is the batch dimension
            and D_IN is the dimension of the feature vector.
        :param y:
            A batch of outputs. The shape is [B, D_OUT], where B is the batch dimension
            and D_OUT is the dimension of the output vector.
        :param layer_1_out:
            The output of the first linear layer, without activation.
        :param layer_2_out:
            The output of the 2nd layer.
        :return:
            A [HIDDEN_DIM] numpy array of weight gradients and a [1] numpy array of bias gradients
        """

        # Calculate the gradients
        # δ2 = δ1 U ◦ (sig(h) * (1 - sig(h))
        # layer_2_grads = get_layer_2_grads(self, X, y, layer_1_out, layer_2_out)

        d_z2 = 2 * (layer_2_out - y)  # derivative of the second linear layer's output = derivative of the loss function
        d_a1 = d_z2 @ self.params_dict['weights_2'].T  # derivative of the first activation function
        d_z1 = d_a1 * self.act_fn(layer_1_out) * (1 - self.act_fn(layer_1_out))  # derivative of the first linear layer's output

        weights1_grad = X.T @ d_z1
        bias1_grad = np.sum(d_z1, axis=0, keepdims=True)

        assert weights1_grad.shape[0] == self.INPUT_DIM
        assert weights1_grad.shape[1] == self.HIDDEN_DIM
        assert weights1_grad.ndim == 2

        assert bias1_grad.shape[0] == 1
        assert bias1_grad.shape[1] == self.HIDDEN_DIM
        assert bias1_grad.ndim == 2

        return weights1_grad, bias1_grad

    def get_layer_2_grads(self, X: np.array, y: np.array,
                          layer_1_out: np.array, layer_2_out: np.array) -> Tuple[
        np.array, np.array]:
        """
               Calculates the gradients for the second layer of the neural network.
           :param X:
               A batch of input feature vectors. The shape is [B, D_IN], where B is the batch dimension
               and D_IN is the dimension of the feature vector.
           :param y:
               A batch of outputs. The shape is [B, D_OUT], where B is the batch dimension
               and D_OUT is the dimension of the output vector.
           :param layer_1_out:
               The output of the first linear layer, without activation.
           :param layer_2_out:
               The output of the 2nd layer.
           :return:
               A [HIDDEN_DIM] numpy array of weight gradients and a [1] numpy array of bias gradients
           """
        weights2_grad = ((layer_2_out - y).T @ self.act_fn(layer_1_out)).T
        bias2_grad = np.sum((layer_2_out - y), keepdims=True)

        assert weights2_grad.ndim == 2
        assert weights2_grad.shape[0] == self.HIDDEN_DIM
        assert weights2_grad.shape[1] == self.OUTPUT_DIM

        assert bias2_grad.shape[0] == 1
        assert bias2_grad.ndim == 2

        return weights2_grad, bias2_grad

    @staticmethod
    def get_loss(y, y_hat) -> np.array:
        """
            Calculate the loss
        :param y:
            The ground-truth responses.
        :param y_hat:
            The predicted responses.
        :return:
            A scalar containing the loss.
        """
        return np.mean((y - y_hat) ** 2)


        # return np.array(0)

    def fit(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array) -> \
    Tuple[List, List]:
        """
            Fits the model to the given data.
        :param X_train:
            The training features
        :param y_train:
            The training outputs
        :param X_test:
            The test features
        :param y_test:
            The test output
        :return:
            The train losses and the test losses
        """
        iter = 0
        train_losses = []
        test_losses = []

        print('X_train shape', X_train.shape, 'y_train shape', y_train.shape)
        print('X_test shape', X_test.shape, 'y_test shape', y_test.shape)

        steps_for_eval = self.MAX_ITER // self.NUM_EVALS

        train_loss = np.inf
        while not self.termination_criteria.is_over(iter, train_loss):
            y_hat = self.forward(X_train)
            train_loss = self.get_loss(y_hat, y_train)

            loss_grad = self.get_loss_grad(X_train, y_train)

            param_update_dict = self.optimizer.step(loss_grad)

            for param_name, param_update in param_update_dict.items():
                self.params_dict[param_name] -= param_update
            
            iter += 1

            if iter % steps_for_eval == 0: 
                y_hat = self.forward(X_test)
                test_loss = self.get_loss(y_hat, y_test)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
              
                print(f'Iteration {iter}, Test Loss={test_loss}')

        return train_losses, test_losses

"""# Generate Data and Train"""

X = np.expand_dims(np.linspace(0, 100, 50), -1)
y = np.expand_dims((5 * X[:, 0] + 3 * X[:, 0] ** 2 + 50), -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nn = NeuralNetSolver()
train_losses, test_losses = nn.fit(X_train, y_train, X_test, y_test)

# plt.plot(len(train_losses))
# plt.plot(train_losses)
plt.plot(test_losses)

"""# Loss Plot"""

plt.plot(range(len(train_losses)), train_losses, 'b')
plt.plot(range(len(train_losses)), test_losses, 'r')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Test Loss'])
plt.xlabel('Step')

"""# Plot of True vs Predicted Values"""

plt.scatter(y, nn.forward(X))
plt.xlabel('True')
plt.ylabel('Predicted')


