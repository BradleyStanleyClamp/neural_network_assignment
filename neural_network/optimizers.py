import torch


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_first_momentum_estimate(self, m_prev, gt):
        """
        mt ← β1 · mt-1 + (1 - β1) · gt (Update biased first moment estimate)

        Arguments:
        m_prev -- previous value of m (biased first momentum term)
        gt -- gradient value e.g dW or db

        Returns:
        m -- biased first momentum estimate
        """
        assert m_prev.shape == gt.shape

        m = self.beta1 * m_prev + (1 - self.beta1) * gt

        assert m.shape == m_prev.shape
        return m

    def calc_unbiased_first_momentum_estimate(self, m, t):
        """
        mbt ← mt / (1 - β1^t) (Compute bias - corrected first moment estimate)

        Arguments:
        m -- biased first momentum estimate
        t -- count of the number of steps taken by Adam

        Returns:
        m_unb -- unbiased first momentum estimate
        """

        m_unb = m / (1 - self.beta1**t)

        assert m_unb.shape == m.shape

        return m_unb

    def update_second_momentum_estimate(self, v_prev, gt):
        """
        vt ← β2 · vt - 1 + (1 - β2) · gt2 (Update biased second raw moment estimate)

        Arguments:
        v_prev -- previous value for the biased second momentum
        gt -- gradient value e.g dW or db

        Returns:
        v -- biased second momentum estimate
        """

        assert v_prev.shape == gt.shape
        v = self.beta2 * v_prev + (1 - self.beta2) * torch.square(gt)

        assert v.shape == v_prev.shape

        return v

    def calc_unbiased_second_momentum_estimate(self, v, t):
        """
        vbt ← vt / (1 - β2^t) (Compute bias - corrected first moment estimate)

        Arguments:
        v -- biased second momentum estimate
        t -- count of the number of steps taken by Adam

        Returns:
        v_unb -- unbiased second momentum estimate
        """

        v_unb = v / (1 - self.beta2**t)
        assert v_unb.shape == v.shape

        return v_unb

    def update_parameters(self, param, m_unb, v_unb):
        """
        θt ← θt - 1 -  ⍺ · mb t /(√vbt + ε) (Update parameters)

        Arguments:
        param -- parameter value to be updated
        m_unb -- unbiased first momentum estimate
        v_unb -- unbiased second momentum estimate

        Returns:
        param_new -- update parameter
        """

        param_new = param - self.learning_rate * (
            m_unb / (torch.sqrt(v_unb) + self.epsilon)
        )

        return param_new

    def optimize_with_adam(self, gt, param, m_prev, v_prev, t):
        """
        Given parameter to update and the gradient of the loss wrt to the parameter, it runs through a full iteration of the adam algorithm and returns the updated values.

        Arguments:
        gt -- gradient of the loss with respect to target parameter
        param -- param to be updated
        m_prev -- previous value of m (biased first momentum term)
        v_prev -- previous value for the biased second momentum
        t -- time step of the Adam process

        Returns:
        param_new -- update parameter
        m -- biased first momentum estimate
        v -- biased second momentum estimate
        """

        m = self.update_first_momentum_estimate(m_prev, gt)
        v = self.update_second_momentum_estimate(v_prev, gt)

        m_unb = self.calc_unbiased_first_momentum_estimate(m, t)
        v_unb = self.calc_unbiased_second_momentum_estimate(v, t)
        # print(f"m_unb: {m_unb}")

        param_new = self.update_parameters(param, m_unb, v_unb)

        return param_new, m, v
