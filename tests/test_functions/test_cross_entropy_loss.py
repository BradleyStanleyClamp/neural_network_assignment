import pytest
import torch
import torch.nn.functional as F
import neural_network as nn


def test_basic_cross_entropy():
    y = torch.ones((4, 1))
    y_hat = torch.ones((4, 1))

    loss = nn.cross_entropy_loss(y_hat, y)

    assert loss == 0


def test_cross_entropy_loss_single_sample():
    """Test cross entropy loss for a single sample."""
    y_hat = torch.tensor([[0.7, 0.2, 0.1]], requires_grad=False)
    y = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=False)
    expected_loss = F.cross_entropy(y_hat.log(), y.argmax(dim=1))
    calculated_loss = nn.cross_entropy_loss(y_hat, y)
    assert torch.isclose(
        calculated_loss, expected_loss
    ), f"Expected {expected_loss}, but got {calculated_loss}"
