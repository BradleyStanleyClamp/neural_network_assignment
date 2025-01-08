import neural_network as nn
import torch


def test_flatten():
    image_tensor = torch.rand(1, 28, 28)

    flattened_tensor = nn.flatten_mnist(image_tensor)

    assert flattened_tensor.size() == torch.Size([784, 1])
