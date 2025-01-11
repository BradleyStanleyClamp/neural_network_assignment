import torch
import pytest

import neural_network as nn


@pytest.mark.parametrize(
    "test_in, test_out, expected_W",
    [
        (1, 1, (1, 1)),
        (4, 8, (8, 4)),
        (246, 64, (64, 246)),
    ],
)
def test_linear_W_init(test_in, test_out, expected_W):
    """
    Testing initialization of linear layer is correct. Expected should be a set of weights and biases of correct dimensions and random values*

    * assuming random weight initialization used
    """

    linlay = nn.Linear_layer(test_in, test_out)

    assert linlay.W.size() == expected_W


@pytest.mark.parametrize(
    "test_in, test_out, expected_b",
    [
        (1, 1, (1, 1)),
        (4, 8, (8, 1)),
        (246, 64, (64, 1)),
    ],
)
def test_linear_b_init(test_in, test_out, expected_b):
    """
    Testing initialization of linear layer is correct. Expected should be a set of weights and biases of correct dimensions and random values*

    * assuming random weight initialization used
    """

    linlay = nn.Linear_layer(test_in, test_out)

    assert linlay.b.size() == expected_b


@pytest.mark.parametrize(
    "size_in, size_out, data_in, expected_out",
    [
        # Test case 1: Single input, single output
        (1, 1, torch.tensor([[2.0]]), torch.tensor([[3.0]])),
        # Test case 2: Single input, multiple outputs
        (1, 2, torch.tensor([[1.0]]), torch.tensor([[2.0], [2.0]])),
        # Test case 3: Multiple inputs, single output
        (2, 1, torch.tensor([[1.0], [2.0]]), torch.tensor([[4.0]])),
        # Test case 4: Multiple inputs, multiple outputs
        (2, 2, torch.tensor([[1.0], [2.0]]), torch.tensor([[4.0], [4.0]])),
    ],
)
def test_linear_forward(size_in, size_out, data_in, expected_out):
    """
    Testing forward pass of linear layer

    """

    linlay = nn.Linear_layer(size_in, size_out, init_mode="ones")

    output = linlay.forward(data_in)
    assert torch.equal(output, expected_out)


# @pytest.mark.parametrize(
#     "size_in, size_out, grad_in, expected_out",
#     [
#         (...),
#         (...),
#         ...
#     ],
# )
# def test_linear_backward(size_in, size_out, data_in, expected_out):
#     """
#     TODO

#     """

#     linlay = nn.Linear_layer(size_in, size_out, init_mode="ones")

#     output = linlay.backward(grad_in)
#     assert torch.equal(output, expected_out)

# def test_linear_forward
