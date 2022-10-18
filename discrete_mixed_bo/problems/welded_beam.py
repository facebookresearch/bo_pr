#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Welded Beam problem from https://link.springer.com/content/pdf/10.1007/s00158-018-2182-1.pdf
"""
from copy import deepcopy
from math import sqrt
from typing import Optional, Tuple

import numpy as np
import torch
from botorch.test_functions.base import ConstrainedBaseTestProblem
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class WeldedBeam(DiscreteTestProblem, ConstrainedBaseTestProblem):
    dim = 6
    _bounds = [
        [0, 1],  # binary, welding type
        [0, 3],  # categorical, metal material
        [0.0625, 2],  # cont/ordinal, thickness of the weld
        [0.1, 10],  # cont/ordinal, length of the welded joint
        [2, 20],  # cont/ordinal, width of the beam
        [0.0625, 2],  # cont/ordinal, thickness of the beam
    ]
    num_constraints = 5
    F = 6000
    delta_max = 0.25
    L = 14

    material_params = {
        # C1, C2, sigma_d, E, G
        0: (0.1047, 0.0481, 3e4, 3e7, 12e6),  # steel
        1: (0.0489, 0.0224, 8e3, 14e6, 6e6),  # cast iron
        2: (0.5235, 0.2405, 5e3, 1e7, 4e6),  # aluminum
        3: (0.5584, 0.2566, 8e3, 16e6, 1e7),  # brass
    }

    # In the discrete setting, the optimal value is 1.9553
    # in the continuous setting, the optimal value is

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        continuous: bool = False,
    ) -> None:
        integer_indices = [0]
        if not continuous:
            self._orig_cont_bounds_list = deepcopy(self._bounds)
            for i in range(2, 6):
                n_ordinal = int((self._bounds[i][1] - self._bounds[i][0]) / 0.0625)
                self._bounds[i][0] = 0
                self._bounds[i][1] = n_ordinal - 1
                integer_indices.append(i)
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            integer_indices=integer_indices,
            categorical_indices=[1],
        )
        self.continuous = continuous
        if not continuous:
            self.register_buffer(
                "_orig_cont_bounds_tensor",
                torch.tensor(self._orig_cont_bounds_list).t(),
            )

    def _split_X(
        self, X: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        X_int = 0.0625 * X[..., 2:].round() + self._orig_cont_bounds_tensor[0, 2:]
        (X2, X3, X4, X5) = torch.split(X_int, 1, dim=-1)
        return X[..., 0], X[..., 1], X2, X3, X4, X5

    def _evaluate_true(self, X):
        params = self._get_params(X=X)
        m = params["m"]
        C1, C2, sigma_d, E, G = self.material_params[int(m.item())]
        t = params["t"]
        l = params["l"]
        return (1 + C1) * (params["w"] * t + l) * params["h"].pow(2) + C2 * t * params[
            "b"
        ] * (self.L + l)

    def _get_params(self, X):

        if not self.continuous:
            w, m, h, l, t, b = self._split_X(X=X)
        else:
            w, m, h, l, t, b = torch.split(X, 1, dim=-1)
        if w == 0:
            A = sqrt(2) * h * l
            J = A * ((h + t).pow(2) / 4 + l.pow(2) / 12)
            R = 0.5 * torch.sqrt(l.pow(2) + (h + t).pow(2))

        elif w == 1:
            A = sqrt(2) * h * (t + l)
            J = sqrt(2) * h * l * ((h + t).pow(2) / 4 + l.pow(2) / 12) + sqrt(
                2
            ) * h * t * ((h + l).pow(2) / 4 + t.pow(2) / 12)
            R = 0.5 * torch.max(
                torch.sqrt(l.pow(2) + (h + t).pow(2)),
                torch.sqrt(t.pow(2) + (h + l).pow(2)),
            )
        else:
            raise ValueError

        cos_theta = l / (2 * R)

        return {
            "w": w,
            "m": m,
            "h": h,
            "l": l,
            "t": t,
            "b": b,
            "A": A,
            "J": J,
            "R": R,
            "cos_theta": cos_theta,
        }

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.stack([self._evaluate_true(x) for x in X.view(-1, 6)], dim=0).view(
            X.shape[:-1]
        )

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        return torch.stack(
            [self._evaluate_slack_true(x) for x in X.view(-1, 6)], dim=0
        ).view(*X.shape[:-1], self.num_constraints)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        params = self._get_params(X=X)
        w = params["w"]
        m = params["m"]
        h = params["h"]
        l = params["l"]
        t = params["t"]
        b = params["b"]
        A = params["A"]
        J = params["J"]
        R = params["R"]
        cos_theta = params["cos_theta"]
        C1, C2, sigma_d, E, G = self.material_params[int(m.item())]
        sigma = 6 * self.F * self.L / (t.pow(2) * b)
        delta = 4 * self.F * self.L**3 / (E * t.pow(3) * b)
        P_c = (
            4.013
            * t
            * b.pow(3)
            * sqrt(E * G)
            / (6 * self.L**2)
            * (1 - t / (4 * self.L) * sqrt(E / G))
        )
        tau_prime = torch.sqrt(self.F / A)
        tau_double_prime = self.F * (self.L + 0.5 * l) * R / J
        tau = torch.sqrt(
            tau_prime.pow(2)
            + tau_double_prime.pow(2)
            + 2 * tau_prime * tau_double_prime * cos_theta
        )
        g1 = 0.577 * sigma_d - tau
        g2 = sigma_d - sigma
        g3 = b - h
        g4 = P_c - self.F
        g5 = self.delta_max - delta
        # positive slack implies feasibility
        return torch.stack([g1, g2, g3, g4, g5], dim=-1)
