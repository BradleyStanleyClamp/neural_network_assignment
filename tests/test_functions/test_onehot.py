import neural_network as nn
import torch
import pytest


def test_one_hot_shape():
    x = 5
    y = nn.one_hot(x)

    assert y.size() == torch.Size([10, 1])


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T),
        (1, torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T),
        (2, torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T),
        (3, torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T),
        (4, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T),
        (5, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]).T),
        (6, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T),
        (7, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]).T),
        (8, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).T),
        (9, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T),
    ],
)
def test_one_hot_functionality(input, expected):
    """
    Testing all possible cases of converting from target value to one hot
    """

    output = nn.one_hot(input)
    print(output)
    print(expected)

    assert torch.equal(output, expected)
