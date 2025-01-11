import torch


def initialize_parameters(input_size, output_size, mode):
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
        biases = torch.zeros(output_size, 1)

    elif mode == "HeInit":
        shape = (input_size, output_size)
        fan_in = output_size
        std = torch.sqrt(torch.tensor(2.0 / fan_in))
        weights = (torch.randn(shape) * std).T
        biases = torch.zeros(output_size, 1)

    elif mode == "ones":
        weights = torch.ones(output_size, input_size)
        biases = torch.ones(output_size, 1)
    else:
        raise NotImplementedError(f"{mode} mode not yet implemented")

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

    # loss = -torch.sum(y * torch.log(clamped_value))
    # m = clamped_value.shape[1]
    # print(f"shape: {m}")
    # print(clamped_value.shape)
    # print(y.shape)

    # loss = (-1 / m) * (
    #     torch.dot(y, torch.log(clamped_value).T)
    #     + torch.dot((1 - y), torch.log(1 - clamped_value).T)
    # )

    loss = -1 * torch.mean(
        (y * torch.log(clamped_value))
    )  # + ((1 - y) * torch.log(clamped_value))
    # )

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


def negative_log_likelihood(predictions, targets):
    """
    Computes the Negative Log-Likelihood (NLL) loss.

    Args:
        predictions (torch.Tensor): Predicted probabilities for each class (shape: [N, C]),
                                    where N is the number of samples and C is the number of classes.
        targets (torch.Tensor): Ground truth class indices (shape: [N]).

    Returns:
        float: The NLL loss.
    """
    # Ensure numerical stability by adding a small value (epsilon) to predictions
    epsilon = 1e-9
    # print(predictions)

    # Gather the predicted probabilities for the true classes
    true_class_probs = predictions[torch.arange(len(targets)), targets]

    # Compute the negative log of the true class probabilities
    nll = -torch.sum(torch.log(true_class_probs)) / len(targets)

    return nll.item()
