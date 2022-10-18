#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Problems with only binary variables.
"""
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


# Code for the contamination problem is adapted from:
# https://github.com/QUVA-Lab/COMBO/blob/master/COMBO/experiments/test_functions/binary_categorical.py.
def generate_contamination_dynamics(dim, random_seed=None):
    n_stages = dim
    n_simulations = 100

    init_alpha = 1.0
    init_beta = 30.0
    contam_alpha = 1.0
    contam_beta = 17.0 / 3.0
    restore_alpha = 1.0
    restore_beta = 3.0 / 7.0
    init_Z = np.random.RandomState(random_seed).beta(
        init_alpha, init_beta, size=(n_simulations,)
    )
    lambdas = np.random.RandomState(random_seed).beta(
        contam_alpha, contam_beta, size=(n_stages, n_simulations)
    )
    gammas = np.random.RandomState(random_seed).beta(
        restore_alpha, restore_beta, size=(n_stages, n_simulations)
    )

    return init_Z, lambdas, gammas


def _contamination(x, dim, cost, init_Z, lambdas, gammas, U, epsilon):
    assert x.size == dim

    rho = 1.0
    n_simulations = 100

    Z = np.zeros((x.size, n_simulations))
    Z[0] = (
        lambdas[0] * (1.0 - x[0]) * (1.0 - init_Z) + (1.0 - gammas[0] * x[0]) * init_Z
    )
    for i in range(
        1,
        dim,
    ):
        Z[i] = (
            lambdas[i] * (1.0 - x[i]) * (1.0 - Z[i - 1])
            + (1.0 - gammas[i] * x[i]) * Z[i - 1]
        )

    below_threshold = Z < U
    constraints = np.mean(below_threshold, axis=1) - (1.0 - epsilon)

    return np.sum(x * cost - rho * constraints)


class Contamination(DiscreteTestProblem):
    """
    Contamination Control Problem.

    The search space consists of only binary variables.
    """

    def __init__(
        self,
        dim: int,
        lamda: float = 0.0,
        noise_std: Optional[float] = None,
        negate: bool = False,
        random_seed: int = 0,
    ) -> None:
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(dim)]
        super().__init__(
            noise_std=noise_std, negate=negate, integer_indices=list(range(dim))
        )
        self.lamda = lamda
        self.init_Z, self.lambdas, self.gammas = generate_contamination_dynamics(
            dim=dim, random_seed=random_seed
        )

    def evaluate_true(self, X: Tensor) -> Tensor:
        res = torch.cat([self._evaluate_single(x) for x in X.view(-1, self.dim)], dim=0)
        return res.view(X.shape[:-1])

    def _evaluate_single(self, x: Tensor) -> Tensor:
        assert x.dim() == 1
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = _contamination(
            x=(x.cpu() if x.is_cuda else x).numpy(),
            dim=self.dim,
            cost=np.ones(x.numel()),
            init_Z=self.init_Z,
            lambdas=self.lambdas,
            gammas=self.gammas,
            U=0.1,
            epsilon=0.05,
        )
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()


class LABS(DiscreteTestProblem):
    """
    Low auto-correlation binary. This problem is adapted from:
    https://github.com/aryandeshwal/MerCBO/blob/main/MerCBO/experiments/test_functions/labs.py
    """

    def __init__(
        self,
        dim: int,
        noise_std: Optional[float] = None,
        negate: bool = False,
        random_seed: int = 0,
    ) -> None:
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(dim)]
        super().__init__(
            noise_std=noise_std, negate=negate, integer_indices=list(range(dim))
        )

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.tensor(
            [self._evaluate_single(x) for x in X.view(-1, self.dim).cpu()],
            dtype=X.dtype,
            device=X.device,
        ).view(X.shape[:-1])

    def _evaluate_single(self, x: Tensor) -> np.ndarray:
        assert x.dim() == 1
        if x.dim() == 2:
            x = x.squeeze(0)
        x = x.numpy()
        N = x.shape[0]
        E = 0
        for k in range(1, N):
            C_k = 0
            for j in range(0, N - k - 1):
                C_k += (-1) ** (1 - x[j] * x[j + k])
            E += C_k**2
        return -1.0 * N / (2 * E)
