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
        # Check multi class problem with one hot vector for truths
        # assert torch.sum(t) == 1

        # Store true labels
        self.t = t

        # Create instance of log softmax class
        softmax = nn.Softmax()

        # Calculate f(s), softmax of the output of model and store them
        self.softmax_probs = softmax.forward(s.T)
        # print(f"self.softmax_probs: {self.softmax_probs}")

        # Check s and t are same shapes
        # assert (
        #     self.softmax_probs.shape == t.shape
        # ), f"Assertion failed: softmax_probs.size = {self.softmax_probs.shape}, t.size = {t.shape}"

        # Calculate loss
        self.loss = nn.negative_log_likelihood(self.softmax_probs, t)

    def backwards(self):
        """
        Calculates the gradients with respect to the outputs of the last linear layer for backpropagation

        Returns:
        dS -- gradients with respect to the outputs of the last linear layer

        """

        # Set output equal to probabilities
        delta = self.softmax_probs.clone()
        # print(f"delta: {delta}")

        # assert self.t.shape == delta.shape

        # Change value for correct class
        # for i, ti in enumerate(self.t):
        #     if ti == 1:
        #         delta[i] -= 1

        # print(f"shape of delta is: {delta.shape}")
        # print(f"t: {self.t}")
        for i in range(len(self.t)):
            # print(self.t[i])
            # print(delta[i, :])
            delta[i, self.t[i]] -= 1
            # print(delta[i, :])

        return delta.T


class Categorical_cross_entropy_TEST:
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
        self.t = t.T

        # Create instance of softmax class
        softmax = nn.Softmax()

        # Calculate f(s), softmax of the output of model and store them
        self.softmax_probs = softmax.forward(s)

        # Check s and t are same shapes
        # assert (
        #     self.softmax_probs.shape == t.shape
        # ), f"Assertion failed: softmax_probs.size = {self.softmax_probs.shape}, t.size = {t.shape}"

        # Calculate loss
        # self.loss = nn.cross_entropy_loss(self.softmax_probs, t)

        loss = torch.nn.CrossEntropyLoss()

        assert (
            s.shape == t.shape
        ), f"Assertion failed: s.size = {s.shape}, t.size = {t.shape}"

        self.input = s.T
        self.input.requires_grad = True

        self.output = loss(self.input, self.t)
        self.loss = self.output.detach().numpy()

        softmax_torch = torch.nn.Softmax()
        nllloss_torch = torch.nn.NLLLoss()

        sft = softmax_torch(self.input)
        print(f"sft: {sft.squeeze()}")
        print(f"self.t: {torch.tensor([torch.argmax(self.t)])}")
        loss_torch = nllloss_torch(sft.squeeze(), torch.tensor([torch.argmax(self.t)]))

        print(f"cross loss: {self.loss}, frac loss: {loss_torch}")

    def backwards(self):
        """
        Calculates the gradients with respect to the outputs of the last linear layer for backpropagation

        Returns:
        dS -- gradients with respect to the outputs of the last linear layer

        """

        self.output.backward()

        assert self.input.is_leaf

        assert self.input.grad.T.shape == torch.Size(
            [10, 1]
        ), f"Assertion failed: grad = {self.input.grad.T.shape}, exp = {torch.Size([10, 1])}"

        return self.input.grad.T


class Categorical_cross_entropy_TEST2:
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
        self.t = t.T

        # Create instance of softmax class
        softmax = nn.Softmax()

        # Calculate f(s), softmax of the output of model and store them
        self.softmax_probs = softmax.forward(s)

        # Check s and t are same shapes
        # assert (
        #     self.softmax_probs.shape == t.shape
        # ), f"Assertion failed: softmax_probs.size = {self.softmax_probs.shape}, t.size = {t.shape}"

        # Calculate loss
        # self.loss = nn.cross_entropy_loss(self.softmax_probs, t)

        loss = torch.nn.CrossEntropyLoss()

        assert (
            s.shape == t.shape
        ), f"Assertion failed: s.size = {s.shape}, t.size = {t.shape}"

        self.input = s.T
        self.input.requires_grad = True

        self.output = loss(self.input, self.t)
        self.loss = self.output.detach().numpy()

        softmax_torch = torch.nn.Softmax()
        nllloss_torch = torch.nn.NLLLoss()

        sft = softmax_torch(self.input)
        loss_torch = nllloss_torch(sft, self.t)

        print(f"cross loss: {self.loss}, frac loss: {loss_torch}")

    def backwards(self):
        """
        Calculates the gradients with respect to the outputs of the last linear layer for backpropagation

        Returns:
        dS -- gradients with respect to the outputs of the last linear layer

        """

        self.output.backward()

        assert self.input.is_leaf

        assert self.input.grad.T.shape == torch.Size(
            [10, 1]
        ), f"Assertion failed: grad = {self.input.grad.T.shape}, exp = {torch.Size([10, 1])}"

        return self.input.grad.T
