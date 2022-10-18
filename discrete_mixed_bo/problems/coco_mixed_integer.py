#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Problems mixed integer continuous search spaces.

References

.. [Hansen2019]
    N. Hansen, D. Brockhoff, O. Mersmann, T. Tusar, D. Tusar, O. A. ElHara, Phillipe R. Sampaio, A. Atamna, K. Varelas, U. Batu, D. M. Nguyen, F. Matzner, A. Auger. COmparing Continuous Optimizers: numbbo/COCO on Github. Zenodo, DOI:10.5281/zenodo.2594848, March 2019.


This code leverages the COCO library (see [Hansen2019]_) and is adapted from
https://github.com/aryandeshwal/HyBO/blob/master/experiments/test_functions/mixed_integer.py
"""

# import cocoex

from typing import Optional

# prepare mixed integer suite
# suite_name = "bbob-mixint"
# output_folder = "cocex-optimize-fmin"
# suite = cocoex.Suite(suite_name, "", "")
import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.sampling import manual_seed
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class BBOBTestFunction(SyntheticTestFunction):
    r"""Base class for BBOB functions.

    d-dimensional function (usually evaluated on the hypercube `[-5, 5]^d`).

    See for [Hansen2019]_ details.
    """

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self._bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        with manual_seed(seed):
            # sample x_opt uniformly in [-4, 4]^d
            self._optimizers = 8 * torch.rand(dim) - 4
            # sample f_opt from a Cauchy distribution with median of 0
            # and roughly 50% of the values between -100 and 100. Clamp
            # the value to be within -1000 and 1000. Round the value to
            # the nearest integer
            self._optimal_value = min(
                max(
                    round(10000 * (torch.randn(1) / torch.randn(1)).item()) / 100,
                    -1000.0,
                ),
                1000.0,
            )
        super().__init__(noise_std=noise_std, negate=negate)


class Sphere(BBOBTestFunction):
    r"""Sphere function.

    d-dimensional function (usually evaluated on the hypercube `[-5, 5]^d`):

    f(x) = \sum_{i=1}^d (x_i - x_{i,opt})^2 + f_opt

    See for [Hansen2019]_ details.
    """

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (X - self.optimizers).pow(2).sum(dim=-1) + self._optimal_value


# class MixedIntegerCOCO(DiscreteTestProblem):
#     """
#     Mixed Integer Black box optimization using cocoex library
#     """
#     def __init__(self, negate: bool = False,noise_std: Optional[float] = None, problem_id: Optional[str] = None) -> None:
#         self.problem_id = problem_id
#         self.problem = suite.get_problem(self.problem_id)
#         self.integer_indices = list(range(self.problem.number_of_integer_variables))
#         self._bounds = list(zip(self.problem.lower_bounds, self.problem.upper_bounds))
#         super().__init__(negate=negate, noise_std=noise_std)

#     def evaluate_true(self, X: Tensor) -> Tensor:
#         X_shape = X.shape
#         if X.dim() > 2:
#             X = X.view(-1, X.shape[-1])
#         return torch.tensor([self.problem(xi) for xi in X.cpu().numpy()], dtype=X.dtype, device=X.device).view(X_shape[:-1])
