from torch import nn
import torch
import torch.optim.adam
from neural_network import (
    Live_plot,
    get_mnist_loaders,
    load_mnist,
    flatten_mnist,
    initialize_parameters,
)

device = "cpu"


class MyOwnNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyOwnNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            ## Softmax layer ignored since the loss function defined is nn.CrossEntropy()
        )
        init_mode = "HeInit"
        fc1W, fc1b = initialize_parameters(784, 128, mode=init_mode)
        fc2W, fc2b = initialize_parameters(128, 64, mode=init_mode)
        fc3W, fc3b = initialize_parameters(64, 10, mode=init_mode)
        # print(self.linear_relu_stack[0].weight.data.shape)
        self.linear_relu_stack[0].weight.data = fc1W
        self.linear_relu_stack[0].bias.data = fc1b.squeeze()
        self.linear_relu_stack[2].weight.data = fc2W
        self.linear_relu_stack[2].bias.data = fc2b.squeeze()
        self.linear_relu_stack[4].weight.data = fc3W
        self.linear_relu_stack[4].bias.data = fc3b.squeeze()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = MyOwnNeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def train_model(model, epochs, train_loader):
    model.train()
    losses = []
    accuracies = []

    live_plot = Live_plot()

    for epoch in range(1, epochs):
        print(f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(train_loader):
            y_pred = model(data)
            loss = loss_fn(y_pred, target)
            optimizer.zero_grad()
            loss.backward()

            print(f"Batch {batch_idx}: Gradients:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad}")
                else:
                    print(f"{name}: No gradient computed")
            assert False
            optimizer.step()

            # Print the values of weights and biases
            # print(f"Batch {batch_idx}: Weights and Biases:")
            # for name, param in model.named_parameters():
            #     print(f"{name} value: {param.data}")

            _correct = (y_pred.argmax(1) == target).type(torch.float).sum().item()
            _batch_size = len(data)

            # print(loss)
            # if batch_idx == 5:
            #     assert False

            batch_accuracy = _correct / _batch_size
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch loss mean: {:.6f}\tBatch accuracy: {:.6f}\t grad: {}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        batch_accuracy,
                        y_pred.grad,
                    )
                )
                losses.append(loss.item())
                accuracies.append(batch_accuracy)
                live_plot.update_plot(losses, accuracies)

    live_plot.keep_plot_showing()


def training_class(model):
    epochs = 4
    batch_size = 64

    train_dataset, test_dataset = load_mnist()

    train_loader, test_loader = get_mnist_loaders(
        train_dataset, test_dataset, batch_size
    )

    train_model(model, epochs, train_loader)


training_class(model)
