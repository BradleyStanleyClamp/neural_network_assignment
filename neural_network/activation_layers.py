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
        self.Z = Z

        assert A.shape == Z.shape

        return A

    def backward(self, dA):
        """
        Argument:
        dA -- Incoming gradient of next layer. Has a size (n_x, 1), where n_x is the number of gradients in the next layer.

        Returns:
        dZ -- output of ReLU derivative. Has size (n_x, 1).
        """

        dZ = dA.clone()

        dZ[self.Z <= 0] = 0

        assert dZ.shape == self.Z.shape

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

        # Subtract max along the last dimension (dim=1) for numerical stability
        Z_max = torch.max(Z, dim=1, keepdim=True).values

        # Compute softmax
        A = torch.exp(Z - Z_max) / torch.sum(torch.exp(Z - Z_max), dim=1, keepdim=True)

        assert A.shape == Z.shape

        return A


class Log_Softmax:
    def __init__(self):
        pass

    def forward(self, Z):
        """
        Argument:
        Z -- input to Log Softmax function, which is the output of the linear part of the layer. Has a size (n_x, batch_size), where n_x is the number of features in the linear part of the layer

        Returns:
        A -- output of Log Softmax function applied to each feature individually. Has size (n_x, batch_size).
        """
        return Z - torch.log(torch.sum(torch.exp(Z))).unsqueeze(-1)
