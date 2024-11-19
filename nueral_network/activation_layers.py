import torch


class ReLU:

    def __init__(self) -> None:
        pass

    def forward(self, x):
        """
        Argument:
        x -- input to ReLU function, which is the output of the linear part of the layer. Has a size (n_x, 1), where n_x is the number of features in the linear part of the layer

        Returns:
        y -- output of ReLU function applied to each feature individually. Has size (n_x, 1).
        """
        y = torch.maximum(x, torch.tensor(0.0))

        return y

    def backward(self, x):
        """
        Argument:
        x -- Incoming gradient of next layer. Has a size (n_x, 1), where n_x is the number of gradients in the next layer.

        Returns:
        y -- output of ReLU derivative. Has size (n_x, 1).
        """

        y = torch.zeros(x.shape)

        for i, xi in enumerate(x):
            if xi > 0:
                y[i] = 1

        return y
