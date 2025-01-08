import torch


def initialize_parameters(input_size, output_size, mode="HeInit"):
    """
    Argument:
    input_size -- number of connections to previous layer
    output_size -- number of neurons in current layer
    mode -- weight initialization mode, default is HeInit

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

    if mode == "HeInit":
        shape = (input_size, output_size)
        fan_in = output_size
        std = torch.sqrt(torch.tensor(2.0 / fan_in))
        weights = (torch.randn(shape) * std).T
        biases = torch.zeros(output_size, 1)

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
    clamped_value = torch.clamp(y_hat, min=1e-19, max=1 - 1e-19)
    # print(f"clamped value: {clamped_value}")
    loss = -torch.sum(y * torch.log(clamped_value))

    assert loss.shape == ()

    return loss


def flatten_mnist(data):
    """
    Flattens mnist image from [1, 28, 28] to [784, 1]

    Arguments
    data -- A torch tensor of size [1, 28, 28], representing a single mnist image

    Returns
    flat -- A torch tensor of size [784, 1]
    """

    return data.view(-1, 1)


def one_hot(target):
    """
    Converts mnist target value into onehot vector

    Arguments
    target -- int value between 0 and 9

    Returns
    target_one_hot -- one hot version of target value
    """

    target_one_hot = torch.zeros([10, 1])

    target_one_hot[target] = 1

    return target_one_hot
