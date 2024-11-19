import nueral_network as nn
import torch

# test1 = nn.Linear_model()

# test1.test_lin_model()


x = torch.rand(4, 1)
x[3] = -5
print(x)

r = nn.ReLU()

print(r.backward(x))
