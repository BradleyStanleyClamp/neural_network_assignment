import torch

import neural_network as nn


class Categorical_cross_entropy:
    def __init__(self, s, t):
        """
        Takes values from last linear layer of network, applies softmax and then cross entropy

        Arguments:
        s -- Output values from last linear layer of the model
        t -- True labels NOT in one hot form

        Returns:
        loss -- cross entropy loss

        """

        # Store true labels
        self.t = t

        # Create instance of log softmax class
        softmax = nn.Softmax()

        # Calculate f(s), softmax of the output of model and store them
        self.softmax_probs = softmax.forward(s.T)

        # Calculate loss
        self.loss = nn.negative_log_likelihood(self.softmax_probs, t)

    def backwards(self):
        """
        Calculates the gradients with respect to the outputs of the last linear layer for backpropagation

        Returns:
        dS -- gradients with respect to the outputs of the last linear layer

        """

        delta = self.softmax_probs.clone()

        for i in range(len(self.t)):
            delta[i, self.t[i]] -= 1

        return delta.T
