from torch import tensor
import torch
from .linear_layer import Linear_layer
from .activation_layers import ReLU
from .functions import flatten_mnist, one_hot
from .categorical_cross_entropy import Categorical_cross_entropy


class Linear_model:

    def __init__(self) -> None:
        init_mode = "HeInit"
        self.fc1 = Linear_layer(784, 128, init_mode=init_mode)
        self.relu1 = ReLU()
        self.fc2 = Linear_layer(128, 64, init_mode=init_mode)
        self.relu2 = ReLU()
        self.fc3 = Linear_layer(64, 10, init_mode=init_mode)

    def forward(self, x):
        x = flatten_mnist(x)
        A = self.relu1.forward(self.fc1.forward(x))
        A = self.relu2.forward(self.fc2.forward(A))
        S = self.fc3.forward(A)

        return S

    def backward(self, dS):
        dA = self.fc3.backward(dS)
        dA = self.fc2.backward(self.relu2.backward(dA))
        dX = self.fc1.backward(self.relu1.backward(dA))

    def update_params(self, learning_rate, optimizer_iteration, optimizer):
        self.fc1.update_params(learning_rate, optimizer_iteration, optimizer)
        self.fc2.update_params(learning_rate, optimizer_iteration, optimizer)
        self.fc3.update_params(learning_rate, optimizer_iteration, optimizer)
