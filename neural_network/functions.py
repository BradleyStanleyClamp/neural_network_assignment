import torch


def initialize_parameters(input_size, output_size, mode="random"):
    """
    Argument:
    input_size -- number of connections to previous layer
    output_size -- number of neurons in current layer
    mode -- weight initialization mode, default is random

    Returns:
    parameters -- python tuple containing:
                    weights -- weight matrix of shape (output_size, input_size)
                    biases -- bias vector of shape (output_size, 1)

    TODO: Initilisation techniqes
    """

    torch.manual_seed(42)

    if mode == "random":
        weights = torch.rand(output_size, input_size)
        biases = torch.rand(output_size, 1)

    elif mode == "ones":
        weights = torch.ones(output_size, input_size)
        biases = torch.ones(output_size, 1)
    else:
        raise NotImplementedError("Other modes not yet implimented")

    return weights, biases


def cross_entropy_loss(y_hat, y):
    """
    CE = - SUM(t_i log(s_i))

    Arguments:
    y_hat -- Output of the model with size (n, 1), where n is the number of neurons in the output layer
    y -- True labels with size (n, 1),  where n is the number of neurons in the output layer

    Returns:
    loss -- cross entropy loss, scalar
    """

    loss = - torch.sum(y * torch.log(y_hat))

    assert loss.shape == ()

    return loss
