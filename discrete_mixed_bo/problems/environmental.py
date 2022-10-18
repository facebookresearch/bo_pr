#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Environmental model calibration problem from:
https://github.com/aryandeshwal/HyBO/blob/master/experiments/test_functions/em_func.py
"""
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


def c(s, t, M, D, L, tau) -> float:
    val = (M * np.exp(-(s**2) / 4 * D * t)) / np.sqrt(4 * np.pi * D * t)
    if t > tau:
        val += (
            (t > tau) * M * np.exp(-((s - L) ** 2) / (4 * D * (t - tau)))
        ) / np.sqrt(4 * np.pi * D * (t - tau))
    return val


def objective(x: np.ndarray) -> float:
    tau = 30.01 + x[0] / 1000
    M = 7 + ((x[1] + 1) * 6) / 2
    D = 0.02 + ((x[2] + 1) * 0.10) / 2
    L = 0.01 + ((x[3] + 1) * 2.99) / 2
    val = 0.0
    for s in [0, 1, 2.5]:
        for t in [15, 30, 45, 60]:
            val += (
                c(s=s, t=t, M=10, D=0.07, L=1.505, tau=30.1525)
                - c(s=s, t=t, M=M, D=D, L=L, tau=tau)
            ) ** 2
    return val


class Environmental(DiscreteTestProblem):
    """
    Environmental model function
    """

    _bounds = [
        (0, 284),
        (-1, 1),
        (-1, 1),
        (-1, 1),
    ]
    dim = 4

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        super().__init__(noise_std=noise_std, negate=negate, integer_indices=[0])

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.tensor(
            [objective(x=x) for x in X.view(-1, self.dim).cpu().numpy()],
            dtype=X.dtype,
            device=X.device,
        ).view(X.shape[:-1])
