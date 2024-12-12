from torch import tensor
from .linear_layer import Linear_layer
from .activation_layers import ReLU


class Linear_model:

    def __init__(self) -> None:
        self.fc1 = Linear_layer(784, 128)
        self.fc2 = Linear_layer(128, 64)
        self.fc3 = Linear_layer(64, 10)

        self.relu = ReLU()

    def forward(self, x):
        x = self.relu.forward(self.fc1.forward(x))
        x = self.relu.forward(self.fc2.forward(x))
        y = self.fc3.forward(x)

        return y

    def test_lin_model(self):
        print(self.fc1.w.shape)
