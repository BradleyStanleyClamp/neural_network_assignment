from torch import tensor
from .linear_layer import Linear_layer


class Linear_model:

    def __init__(self) -> None:
        self.fc1 = Linear_layer(3 * 32 * 32, 128)

    def test_lin_model(self):
        print(self.fc1.w.shape)
