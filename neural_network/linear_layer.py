import torch
from .functions import initialize_parameters


class Linear_layer:

    def __init__(self, input_size, output_size, init_mode="random") -> None:
        """
        Arguments:
        input_size -- size of input layer
        output_size -- size of output of the layer (number of neurons)

        Instance variables:
        self.W -- Weight matrix of size (output_size, input_size)
        self.b -- Bias matrix of size (output_size, 1)
        """
        self.W, self.b = initialize_parameters(input_size, output_size, mode=init_mode)

    def forward(self, x):
        """
        Arguments:
        x -- input vector from previous layer / data input of size (n_x, 1), where n_x is the number of features

        Returns:
        z -- output of linear layer defined by z = Wx + b

        TODO: Cache relevant data
        """

        z = torch.matmul(self.W, x) + self.b

        assert z.shape == (self.W.shape[0], x.shape[1])

        return z
    
    def backward(self, dZ):
        """
        Arguments:
        dZ -- Gradient of the cost wrt the output of this layer

        Returns:
        dA_prev -- Gradients of the cost wrt the activation of the prev layer 
        
        """
