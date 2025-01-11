import torch
from .functions import initialize_parameters
from .optimizers import Adam


class Linear_layer:

    def __init__(self, input_size, output_size, init_mode) -> None:
        """
        Arguments:
        input_size -- size of input layer
        output_size -- size of output of the layer (number of neurons)

        Instance variables:
        self.W -- Weight matrix of size (output_size, input_size)
        self.b -- Bias matrix of size (output_size, 1)
        """
        self.W, self.b = initialize_parameters(input_size, output_size, mode=init_mode)

        # For adam optimizer
        self.Wm = torch.zeros(self.W.shape)
        self.Wv = torch.zeros(self.W.shape)
        self.bm = torch.zeros(self.b.shape)
        self.bv = torch.zeros(self.b.shape)

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
        # m = dZ.shape[0]

        # self.dW = (1 / m) * torch.matmul(dZ, self.A_prev.T)
        # self.db = torch.ones(dZ.shape) * torch.mean(dZ)
        # dA_prev = torch.matmul(self.W.T, dZ)

        m = self.A_prev.shape[1]
        self.dW = (1 / m) * torch.matmul(dZ, self.A_prev.T)
        self.db = (1 / m) * torch.sum(dZ, axis=1, keepdims=True)
        dA_prev = torch.matmul(self.W.T, dZ)

        # print(f"dW: {self.dW}")
        # print(f"db: {self.db}")

        assert dA_prev.shape == self.A_prev.shape
        assert self.dW.shape == self.W.shape
        assert self.db.shape == self.b.shape

        return dA_prev

    def update_params(
        self, learning_rate, optimizer_iteration, optimizer="gradient decent"
    ):
        if optimizer == "gradient decent":
            self.W = self.W - learning_rate * self.dW
            self.b = self.b - learning_rate * self.db

        elif optimizer == "adam":
            adam = Adam()

            # Update weights
            self.W, self.Wm, self.Wv = adam.optimize_with_adam(
                self.dW, self.W, self.Wm, self.Wv, optimizer_iteration
            )

            # Update Biases
            self.b, self.bm, self.bv = adam.optimize_with_adam(
                self.db, self.b, self.bm, self.bv, optimizer_iteration
            )
