#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Problems with only binary variables.
"""
from math import pi
from typing import Optional, Tuple

import numpy as np
import torch
from botorch.test_functions.base import ConstrainedBaseTestProblem
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class PressureVessel(DiscreteTestProblem, ConstrainedBaseTestProblem):
    dim = 4
    _bounds = [
        (1, 100),  # integer
        (1, 100),  # integer
        (10, 200),  # continuous
        (10, 240),  # continuous
    ]
    num_constraints = 3

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            integer_indices=[0, 1],
        )

    @staticmethod
    def _split_X(X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        X_int = 0.0625 * X[..., :2].round()
        x1 = X_int[..., 0]
        x2 = X_int[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        return x1, x2, x3, x4

    def evaluate_true(self, X):
        x1, x2, x3, x4 = self._split_X(X=X)
        return (
            (0.6224 * x1 * x3 * x4)
            + (1.7781 * x2 * x3 * x3)
            + (3.1661 * x1 * x1 * x4)
            + (19.84 * x1 * x1 * x3)
        )

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        # positive slack implies feasibility
        x1, x2, x3, x4 = self._split_X(X=X)
        g1 = x1 - (0.0193 * x3)
        g2 = x2 - (0.00954 * x3)
        g3 = pi * x3 * x3 * x4 + 4.0 / 3.0 * pi * x3 * x3 * x3 - 1296000
        return torch.stack([g1, g2, g3], dim=-1)
