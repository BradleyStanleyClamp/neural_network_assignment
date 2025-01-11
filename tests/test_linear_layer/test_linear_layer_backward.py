import torch
import pytest
import numpy as np

import neural_network as nn


def test_linear_backwards():
    np.random.seed(1)
    dZ_np = np.random.randn(3, 4)
    A_prev_np = np.random.randn(5, 4)
    W_np = np.random.randn(3, 5)
    b_np = np.random.randn(3, 1)

    dZ = torch.tensor(dZ_np, dtype=torch.float32)
    A_prev = torch.tensor(A_prev_np, dtype=torch.float32)
    W = torch.tensor(W_np, dtype=torch.float32)
    b = torch.tensor(b_np, dtype=torch.float32)

    # m = dZ.shape[0]
    m = A_prev.shape[1]

    # dW = (1 / m) * torch.matmul(dZ, A_prev.T)
    print(f"torch dZ: {dZ.shape}")
    print(f"np dZ: {dZ_np.shape}")
    print(f"torch A_prev: {A_prev.shape}")
    print(f"np A_prev: {A_prev_np.shape}")

    dW = (1 / m) * torch.matmul(dZ, A_prev.T)
    db = (1 / m) * torch.sum(dZ, axis=1, keepdims=True)
    dA_prev = torch.matmul(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    #     dA_prev =
    #  [[-1.15171336  0.06718465 -0.3204696   2.09812712]
    #  [ 0.60345879 -3.72508701  5.81700741 -3.84326836]
    #  [-0.4319552  -1.30987417  1.72354705  0.05070578]
    #  [-0.38981415  0.60811244 -1.25938424  1.47191593]
    #  [-2.52214926  2.67882552 -0.67947465  1.48119548]]
    # dW =
    #  [[ 0.07313866 -0.0976715  -0.87585828  0.73763362  0.00785716]
    #  [ 0.85508818  0.37530413 -0.59912655  0.71278189 -0.58931808]
    #  [ 0.97913304 -0.24376494 -0.08839671  0.55151192 -0.10290907]]
    # db =
    #  [[-0.14713786]
    #  [-0.11313155]
    #  [-0.13209101]]

    print(f"dA_prev: {dA_prev}")
    print(f"dW: {dW}")
    print(f"db: {db}")


test_linear_backwards()
