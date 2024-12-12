import pytest
import torch
import neural_network as nn

# Instantiate your softmax function


def test_softmax_basic():
    """Test softmax with a basic tensor."""
    softmax = nn.Softmax()
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    expected_output = torch.tensor([0.0900, 0.2447, 0.6652])  # Manually calculated
    output = softmax.forward(input_tensor)
    assert torch.allclose(
        output, expected_output, atol=1e-4
    ), f"Output mismatch: {output}"


def test_softmax_negative():
    """Test softmax with negative values."""
    softmax = nn.Softmax()
    input_tensor = torch.tensor([-1.0, -2.0, -3.0])
    expected_output = torch.tensor([0.6652, 0.2447, 0.0900])  # Manually calculated
    output = softmax.forward(input_tensor)
    assert torch.allclose(
        output, expected_output, atol=1e-4
    ), f"Output mismatch: {output}"


def test_softmax_zero():
    """Test softmax with zeros."""
    softmax = nn.Softmax()
    input_tensor = torch.tensor([0.0, 0.0, 0.0])
    expected_output = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    output = softmax.forward(input_tensor)
    assert torch.allclose(
        output, expected_output, atol=1e-4
    ), f"Output mismatch: {output}"


def test_softmax_large_numbers():
    """Test softmax with large numbers to check numerical stability."""
    softmax = nn.Softmax()
    input_tensor = torch.tensor([1000.0, 1001.0, 1002.0])
    expected_output = torch.tensor([0.0900, 0.2447, 0.6652])  # Using offset trick
    output = softmax.forward(input_tensor)
    assert torch.allclose(
        output, expected_output, atol=1e-4
    ), f"Output mismatch: {output}"



def test_softmax_invalid_input():
    """Test softmax with invalid input (non-tensor)."""
    softmax = nn.Softmax()
    with pytest.raises(TypeError):
        softmax.forward([1.0, 2.0, 3.0])  # List instead of tensor

    with pytest.raises(TypeError):
        softmax.forward("invalid input")



def test_softmax_single_element():
    """Test softmax with a tensor containing a single element."""
    softmax = nn.Softmax()
    input_tensor = torch.tensor([5.0])
    expected_output = torch.tensor([1.0])
    output = softmax.forward(input_tensor)
    assert torch.equal(output, expected_output), f"Output mismatch: {output}"
