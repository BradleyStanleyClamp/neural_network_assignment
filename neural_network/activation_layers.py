import torch


class ReLU:

    def __init__(self) -> None:
        pass

    def forward(self, Z):
        """
        Argument:
        Z -- input to ReLU function, which is the output of the linear part of the layer. Has a size (n_x, 1), where n_x is the number of features in the linear part of the layer

        Returns:
        A -- output of ReLU function applied to each feature individually. Has size (n_x, 1).
        """
        A = torch.maximum(Z, torch.tensor(0.0))

        return Z

    def backward(self, dA):
        """
        Argument:
        dA -- Incoming gradient of next layer. Has a size (n_x, 1), where n_x is the number of gradients in the next layer.

        Returns:
        dZ -- output of ReLU derivative. Has size (n_x, 1).
        """

        dZ = dA.clone()

        dZ[dA <= 0] = 0

        return dZ


class Softmax:

    def __init__(self) -> None:
        pass

    def forward(self, Z):
        """
        Argument:
        Z -- input to Softmax function, which is the output of the linear part of the layer. Has a size (n_x, 1), where n_x is the number of features in the linear part of the layer

        Returns:
        A -- output of Softmax function applied to each feature individually. Has size (n_x, 1).
        """

        A = torch.divide(
            torch.exp(Z - torch.max(Z)), torch.sum(torch.exp(Z - torch.max(Z)))
        )

        assert A.shape == Z.shape

        return A
