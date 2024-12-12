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
        biases = torch.zeros(output_size, 1)

    else:
        raise NotImplementedError("Other modes not yet implimented")

    return weights, biases


def cross_entropy_loss(y_hat, y):
    """
    Arguments:
    y_hat -- Output of the model
    y -- True labels

    Returns:
    loss -- cross entropy loss
    """

    m = y.shape[1]

    loss = (-1 / m) * (
        torch.dot(y, torch.log(y_hat).T) + torch.dot((1 - y), torch.log(1 - y_hat).T)
    )

    loss = torch.squeeze(loss)
    print(loss.shape)
    assert loss.shape == ()

    return loss
