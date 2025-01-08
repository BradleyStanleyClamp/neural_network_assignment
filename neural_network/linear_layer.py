import torch
from .functions import initialize_parameters


class Linear_layer:

    def __init__(self, input_size, output_size, init_mode="HeInit") -> None:
        """
        Arguments:
        input_size -- size of input layer
        output_size -- size of output of the layer (number of neurons)

        Instance variables:
        self.W -- Weight matrix of size (output_size, input_size)
        self.b -- Bias matrix of size (output_size, 1)
        """
        self.W, self.b = initialize_parameters(input_size, output_size, mode=init_mode)

    def forward(self, A_prev):
        """
        Arguments:
        A_prev -- input vector from previous layer / data input of size (n_x, 1), where n_x is the number of features

        Returns:
        Z -- output of linear layer defined by Z = W A_Prev + b

        """

        # 'Cache' A_prev as is needed for backprop
        self.A_prev = A_prev

        Z = torch.matmul(self.W, A_prev) + self.b

        assert Z.shape == (self.W.shape[0], A_prev.shape[1])

        return Z

    def backward(self, dZ):
        """
        Arguments:
        dZ -- Gradient of the cost wrt the output of this linear layer

        Returns:
        dA_prev -- Gradients of the cost wrt the activation of the prev layer
        dW -- Gradient of the cost wrt weights of current layer l, same shape as self.W
        db -- Gradient of the cost wrt biases current layer l, same shape as self.b

        """
        m = dZ.shape[0]

        self.dW = (1 / m) * torch.matmul(dZ, self.A_prev.T)
        self.db = torch.ones(dZ.shape) * torch.mean(dZ)
        dA_prev = torch.matmul(self.W.T, dZ)

        assert dA_prev.shape == self.A_prev.shape
        assert self.dW.shape == self.W.shape
        assert self.db.shape == self.b.shape

        return dA_prev

    def update_params(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
