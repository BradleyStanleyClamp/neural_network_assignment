import torch

import neural_network as nn


class Categorical_cross_entropy:
    def __init__(self, s, t):
        """
        Takes values from last linear layer of network, applies softmax and then cross entropy

        Arguments:
        s -- Output values from last linear layer of the model
        t -- True labels

        Returns:
        loss -- cross entropy loss

        """
        # Check multi class problem with one hot vector for truths
        assert torch.sum(t) == 1

        # Store true labels
        self.t = t

        # Create instance of softmax class
        softmax = nn.Softmax()

        # Calculate f(s), softmax of the output of model and store them
        self.softmax_probs = softmax.forward(s)

        # Check s and t are same shapes
        assert (
            self.softmax_probs.shape == t.shape
        ), f"Assertion failed: softmax_probs.size = {self.softmax_probs.shape}, t.size = {t.shape}"

        # Calculate loss
        self.loss = nn.cross_entropy_loss(self.softmax_probs, t)

    def backwards(self):
        """
        Calculates the gradients with respect to the outputs of the last linear layer for backpropagation

        Returns:
        dS -- gradients with respect to the outputs of the last linear layer

        """

        # Set output equal to probabilities
        delta = self.softmax_probs

        # Change value for correct class
        for i, ti in enumerate(self.t):
            if ti == 1:
                delta[i] -= 1

        return delta
