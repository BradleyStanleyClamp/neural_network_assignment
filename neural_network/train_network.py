import torch
from .linear_model import Linear_model
from .functions import one_hot
from .categorical_cross_entropy import (
    Categorical_cross_entropy,
)
from .data_preproc import load_mnist, get_mnist_loaders
import matplotlib.pyplot as plt


class Train_network:
    """
    Class for training Linear model. Consists of hierarchical methods for epoch, batch, data point as well as visualization and storage techniques
    """

    def __init__(self, dataset, learning_rate, optimizer, batch_size, epochs, model):
        """
        Initialise class with everything needed for training.
        -- dataset to load in form of string
        -- learning rate
        -- batch size

        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.optimizer_iterator = 1

        if dataset == "mnist":
            self.train_dataset, self.test_dataset = load_mnist()
            self.train_loader, self.test_loader = get_mnist_loaders(
                self.train_dataset, self.test_dataset, batch_size=batch_size
            )

    def train_model(self, live_plot_bool=False, test_on_data=False):
        """
        High level method for training the model
        """

        self.losses = []
        self.accuracies = []

        if live_plot_bool:
            self.live_plot = Live_plot()

        for epoch in range(1, self.epochs):
            if test_on_data:
                self.train_on_single_data_point_test(epoch, live_plot_bool)

            else:
                self.train_single_epoch(epoch, live_plot_bool)

        if live_plot_bool:
            self.live_plot.keep_plot_showing()

    def train_single_epoch(self, epoch, live_plot_bool):
        """
        Method for training Linear Model for a single Epoch.
        - Assumes using mnist data

        """
        for batch_idx, (data, target) in enumerate(self.train_loader):

            S = self.model.forward(data)
            loss = Categorical_cross_entropy(S, target)

            max_indices = torch.argmax(loss.softmax_probs, dim=1)
            comparison = max_indices == target
            correct_count = comparison.sum().item()
            batch_accuracy = correct_count / len(data)

            dS = loss.backwards()
            assert dS.shape == S.shape
            self.model.backward(dS)

            self.model.update_params(
                self.learning_rate, self.optimizer_iterator, self.optimizer
            )
            self.optimizer_iterator += 1
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch loss mean: {:.6f}\tBatch accuracy: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.loss,
                        batch_accuracy,
                    )
                )
                self.losses.append(loss.loss)
                self.accuracies.append(batch_accuracy)
                if live_plot_bool:
                    self.live_plot.update_plot(self.losses, self.accuracies)


class Live_plot:
    def __init__(self):
        """
        Setups figure for live plotting

        Return:
        fig, ax
        """

        plt.ion()
        self.fig, self.axs = plt.subplots(1, 2)

        for ax in self.axs:
            ax.grid()
            ax.set_xlabel("Iteration")
        self.axs[0].set_ylabel("Loss")
        self.axs[0].set_title("Loss vs Iteration")
        self.axs[1].set_ylabel("Accuracy")
        self.axs[1].set_title("Accuracy vs Iteration")
        (self.line,) = self.axs[0].plot([], [], "r-")
        (self.acc,) = self.axs[1].plot([], [], "b-")

    def update_plot(self, losses, accuracies):
        self.line.set_xdata(range(len(losses)))
        self.line.set_ydata(losses)

        self.acc.set_xdata(range(len(losses)))
        self.acc.set_ydata(accuracies)

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        plt.draw()
        plt.pause(0.01)

    def keep_plot_showing(self):
        plt.ioff()
        plt.show()
