import torch
import neural_network as nn


def loss_function(m):
    return m**2 - 2 * m + 1


def grad_function(m):
    return 2 * m - 2


def check_convergence(w0, w1):
    return w0.item() == w1.item()


def test_adam_1D():
    w0 = torch.tensor([0])
    b0 = torch.tensor([0])

    wm = torch.tensor([0])
    wv = torch.tensor([0])

    bm = torch.tensor([0])
    bv = torch.tensor([0])

    adam = nn.Adam()
    t = 1

    converged = False

    while not converged:
        dw = grad_function(w0)
        db = grad_function(b0)
        # print(f"dw: {dw}")

        w0_old = w0
        w0, wm, wv = adam.optimize_with_adam(dw, w0, wm, wv, t)

        # b0, bm, bv = adam.optimize_with_adam(db, b0, bm, bv, t)

        if check_convergence(w0, w0_old):
            # print(f"converged after {t} iterations")
            converged = True

        else:
            # print(f"Iteration {t}: weight: {w0}")
            t += 1

    assert converged


def loss_function_2D(x, y):
    return x**2 + y**2


def grad_function_2D(x):

    return torch.tensor([x[0] * 2, x[1] * 2])
