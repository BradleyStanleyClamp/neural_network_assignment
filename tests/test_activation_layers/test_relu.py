import pytest
import torch
import neural_network as nn

# Instantiate your ReLU function
relu = nn.ReLU()


def test_relu_forward_basic():
    """Test ReLU forward pass with basic input."""
    input_tensor = torch.tensor([[-1.0, 2.0, 3.0, -4.0]]).T
    expected_output = torch.tensor([[0.0, 2.0, 3.0, 0.0]]).T
    output = relu.forward(input_tensor)
    assert torch.equal(
        output, expected_output
    ), f"Output mismatch: {output}, {expected_output}"


def test_relu_forward_zero():
    """Test ReLU forward pass with zeros."""
    input_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    expected_output = input_tensor
    output = relu.forward(input_tensor)
    assert torch.equal(output, expected_output), f"Output mismatch: {output}"


def test_relu_forward_large():
    """Test ReLU forward pass with large positive values."""
    input_tensor = torch.tensor([[1000.0, -1000.0], [500.0, -500.0]])
    expected_output = torch.tensor([[1000.0, 0.0], [500.0, 0.0]])
    output = relu.forward(input_tensor)
    assert torch.equal(output, expected_output), f"Output mismatch: {output}"


def test_relu_backward_basic():
    """Test ReLU backward pass with basic input."""
    forward_input = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
    gradient_output = torch.tensor(
        [[0.5, -0.5], [-0.5, 0.5]]
    )  # Mock gradient for testing
    expected_gradient = torch.tensor(
        [[0.5, 0.0], [0.0, 0.5]]
    )  # Gradient only flows through positive values
    relu.forward(forward_input)  # Simulate the forward pass before backward
    backward_output = relu.backward(gradient_output)
    assert torch.equal(
        backward_output, expected_gradient
    ), f"Output mismatch: {backward_output}"


def test_relu_backward_zero():
    """Test ReLU backward pass with zero input."""
    gradient_output = torch.tensor([[0.0, -1.0], [-5.0, 1.0]])
    expected_gradient = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    backward_output = relu.backward(gradient_output)
    assert torch.equal(
        backward_output, expected_gradient
    ), f"Output mismatch: {backward_output}"


def test_relu_backward_all_positive():
    """Test ReLU backward pass with all positive forward input."""
    forward_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gradient_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected_gradient = gradient_output
    relu.forward(forward_input)
    backward_output = relu.backward(gradient_output)
    assert torch.equal(
        backward_output, expected_gradient
    ), f"Output mismatch: {backward_output}"


def test_relu_empty_tensor():
    """Test ReLU forward and backward with an empty tensor."""
    input_tensor = torch.tensor([])
    gradient_tensor = torch.tensor([])
    forward_output = relu.forward(input_tensor)
    backward_output = relu.backward(gradient_tensor)
    assert torch.equal(
        forward_output, input_tensor
    ), f"Forward mismatch: {forward_output}"
    assert torch.equal(
        backward_output, gradient_tensor
    ), f"Backward mismatch: {backward_output}"
