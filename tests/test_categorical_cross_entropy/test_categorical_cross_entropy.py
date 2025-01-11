import torch
from neural_network import (
    Log_Softmax,
    Softmax,
    Categorical_cross_entropy,
    negative_log_likelihood,
)


# Input
torch.manual_seed(42)
input1 = torch.randn(1, 5, requires_grad=True)
print(f"input1: {input1}")

# Targets
target1 = torch.tensor([2])
target2 = torch.tensor([[0, 0, 1, 0, 0]])
print(f"target1: {target1}")

# My broken code
my_softmax = Softmax()
my_loss = negative_log_likelihood(my_softmax.forward(input1), target1)
print(f"my loss: {my_loss}")

# My cre
my_cre = Categorical_cross_entropy(input1, target1)
print(f"my cre: {my_cre.loss}")
my_dS = my_cre.backwards()
print(f"my_ds: {my_dS}")

# Torch cross entropy
loss1 = torch.nn.CrossEntropyLoss()
torch_cre_loss = loss1(input1, target1)

print(f"torch_cre_loss: {torch_cre_loss}")

torch_cre_loss.backward()
print(f"torch cre dS: {input1.grad}")


# their log soft and nlll
log_softmax = torch.nn.LogSoftmax(dim=1)

loss_fn = torch.nn.NLLLoss()

torch_comb_loss = loss_fn(log_softmax(input1), target1)
print(f"torch_comb_loss: {torch_comb_loss}")

# loss.backward()
