import neural_network as nn
import torch


x = torch.ones((784, 1))
y = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T

model = nn.Linear_model()

y_hat = model.forward(x)


print(y_hat.shape)
print(y.shape)
# assert y_hat.shape == y.shape

y_hat = torch.tensor([1.0]).T
y = torch.tensor([2.0]).T


loss = nn.cross_entropy_loss(y_hat, y)

# model.backwards(loss)
