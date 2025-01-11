from torch import tensor
import torch
from .linear_layer import Linear_layer
from .activation_layers import ReLU
from .functions import flatten_mnist, one_hot
from .categorical_cross_entropy import Categorical_cross_entropy


class Linear_model:

    def __init__(self) -> None:
        init_mode = "HeInit"
        self.fc1 = Linear_layer(784, 128, init_mode=init_mode)
        self.relu1 = ReLU()
        self.fc2 = Linear_layer(128, 64, init_mode=init_mode)
        self.relu2 = ReLU()
        self.fc3 = Linear_layer(64, 10, init_mode=init_mode)

    def forward(self, x):
        x = flatten_mnist(x)
        A = self.relu1.forward(self.fc1.forward(x))
        A = self.relu2.forward(self.fc2.forward(A))
        S = self.fc3.forward(A)

        return S

    def backward(self, dS):
        dA = self.fc3.backward(dS)
        dA = self.fc2.backward(self.relu2.backward(dA))
        dX = self.fc1.backward(self.relu1.backward(dA))

    def update_params(self, learning_rate, optimizer_iteration, optimizer):
        self.fc1.update_params(learning_rate, optimizer_iteration, optimizer)
        self.fc2.update_params(learning_rate, optimizer_iteration, optimizer)
        self.fc3.update_params(learning_rate, optimizer_iteration, optimizer)

    def train_model(self, epoch, train_loader, learning_rate):
        """
        OBSOLETE
        Trains model using stochastic gradient decent

        """

        # Iterate through batches, currently a batch contains a single image
        for batch_idx, (data, target) in enumerate(train_loader):

            # if batch_idx == 2:
            #     return

            sum_dS = 0
            # loss_sum = 0
            batch_size = data.shape[0]
            for item_indx in range(0, batch_size):
                data_point = data[item_indx, :, :, :]
                target_point = target[item_indx]

                # converting target into one hot
                target_one_hot = one_hot(target_point)

                # Forward pass of data
                S = self.forward(data_point)
                # print(f"S: {S.T}")
                # print(f"fc3 weights: {self.fc3.W[0,0]}")

                # Calculate the categorical cross entropy loss
                loss = Categorical_cross_entropy(S, target_one_hot)

                # print(f"softmax probs: {loss.softmax_probs.T}")
                # loss_sum += loss.loss
                # Find the gradient of the loss wrt to the output
                dS = loss.backwards()
                # print(f"dS: {dS.T}")
                sum_dS += dS

            # print(sum_dS)
            avg_dS = sum_dS / batch_size

            # Back propagate through the network
            self.backward(avg_dS)

            # Update the parameters
            self.update_params(learning_rate)
            # print(f"updated fc3 weights: {self.fc3.W[0, 0]}")

            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.loss,
                    )
                )

        # return loss_sum / len(train_loader.dataset)
