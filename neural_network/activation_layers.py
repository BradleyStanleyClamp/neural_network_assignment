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

        y = x.clone()

        y[x <= 0] = 0

        return y


class Softmax:

    def __init__(self) -> None:
        pass

    def forward(self, x):
        """
        Argument:
        x -- input to Softmax function, which is the output of the linear part of the layer. Has a size (n_x, 1), where n_x is the number of features in the linear part of the layer

        Returns:
        y -- output of Softmax function applied to each feature individually. Has size (n_x, 1).
        """

        y = torch.divide(
            torch.exp(x - torch.max(x)), torch.sum(torch.exp(x - torch.max(x)))
        )

        assert y.shape == x.shape

        return y

    def backward(self, x, loss_grad):
        """
        Argument:
        x -- input to Softmax function, which is the output of the linear part of the layer. Has a size (n_x, 1), where n_x is the number of features in the linear part of the layer
        loss_grad -- backpropagated loss gradient value with size (n_x, 1)

        Returns:
        dZ -- gradient of cost wrt to Z.
        """

        # TODO: Implement ... ?
