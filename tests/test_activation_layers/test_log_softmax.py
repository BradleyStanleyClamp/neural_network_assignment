import torch
import neural_network as nn


def test_log_softmax():
    input = torch.tensor([[0.3452, -0.0267, 0.4066]]).T
    print(input.shape)

    softmax = nn.Softmax()
    log_softmax = nn.Log_Softmax()

    out = softmax.forward(input)
    log_out = log_softmax.forward(input)

    print(out)
    print(log_out)


test_log_softmax()
