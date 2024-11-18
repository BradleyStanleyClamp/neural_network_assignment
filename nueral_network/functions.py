import torch


def initialize_parameters(input_size, output_size, mode="random"):
    """
    Argument:
    input_size -- number of connections to previous layer
    output_size -- number of neurons in current layer
    mode -- weight initialization mode, default is random

    Returns:
    parameters -- python tuple containing:
                    weights -- weight matrix of shape (input_size, output_size)
                    biases -- bias vector of shape (output_size, 1)
    """

    torch.manual_seed(42)

    if mode == "random":
        weights = torch.rand(input_size, output_size)
        biases = torch.rand(output_size, 1)

    else:
        assert "Other modes not yet implimented"

    return weights, biases
