#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
3-Objective Electrospun Oil Sorbent optimization problem

References

.. [Wang2020]
    B. Wang, J. Cai, C. Liu, J. Yang, X. Ding. Harnessing a Novel Machine-Learning-Assisted Evolutionary Algorithm to Co-optimize Three Characteristics of an Electrospun Oil Sorbent. ACS Applied Materials & Interfaces, 2020.
"""
from typing import List, Optional

import numpy as np
import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.torch import BufferDict
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class OilSorbent(DiscreteTestProblem, MultiObjectiveTestProblem):
    """All objectives should be minimized.

    The reference point comes from using the infer_reference_point
    method on the full discrete search space.
    """

    # _max_hv = 1177461.0 # full discrete case
    _max_hv = 1279774.75
    _discrete_values = {
        # "V1": [3 / 7, 4 / 6, 1, 6 / 4, 7 / 3],
        "V2": [0.7, 1, 1.4, 1.7, 2],
        "V3": [12, 15, 18, 21, 24],
        "V4": [0.12, 0.135, 0.15, 0.165, 0.18],
        # "V5": [0, 0.04, 0.08, 0.10, 0.20],
        "V6": [16, 20, 26, 28],
        "V7": [0.41, 0.6, 0.84, 1.32],
    }

    _bounds = [
        (0, 1),  # continuous
        (0, 4),  # 5 ordinal values
        (0, 4),  # 5 ordinal values
        (0, 4),  # 5 ordinal values
        (0, 1),  # continuous
        (0, 3),  # 4 ordinal values
        (0, 3),  # 4 ordinal values
    ]
    dim = 7
    num_objectives = 3
    # _ref_point = [-133.9736, -4.8289, 38.6565] # full discrete case
    _ref_point = [-125.3865, -57.8292, 43.2665]

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        integer_indices: Optional[List[int]] = None,
    ) -> None:
        if integer_indices is None:
            integer_indices = [1, 2, 3, 5, 6]
        MultiObjectiveTestProblem.__init__(
            self,
            noise_std=noise_std,
            negate=negate,
        )
        self._setup(integer_indices=integer_indices)
        self.discrete_values = BufferDict()
        for k, v in self._discrete_values.items():
            self.discrete_values[k] = torch.tensor(v, dtype=torch.float)
            self.discrete_values[k] /= self.discrete_values[k].max()

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_split = list(torch.split(X, 1, -1))
        # remap from integer space to proper space
        for i, V_i in enumerate(X_split):
            name = f"V{i+1}"
            if name in self.discrete_values:

                X_split[i] = self.discrete_values[name][V_i.view(-1).long()].view(
                    V_i.shape
                )
        V1, V2, V3, V4, V5, V6, V7 = X_split
        wca = (
            -197.0928
            - 78.3309 * V1
            + 98.6355 * V2
            + 300.0701 * V3
            + 89.8360 * V4
            + 208.2343 * V5
            + 332.9341 * V6
            + 135.6621 * V7
            - 11.0715 * V1 * V2
            + 201.8934 * V1 * V3
            + 17.1270 * V1 * V4
            + 2.5198 * V1 * V5
            - 109.3922 * V1 * V6
            + 30.1607 * V1 * V7
            - 46.1790 * V2 * V3
            + 19.2888 * V2 * V4
            - 102.9493 * V2 * V5
            - 19.1245 * V2 * V6
            + 53.6297 * V2 * V7
            - 73.0649 * V3 * V4
            - 37.7181 * V3 * V5
            - 219.1268 * V3 * V6
            - 55.3704 * V3 * V7
            + 3.8778 * V4 * V5
            - 6.9252 * V4 * V6
            - 105.1650 * V4 * V7
            - 34.3181 * V5 * V6
            - 36.3892 * V5 * V7
            - 82.3222 * V6 * V7
            - 16.7536 * V1.pow(2)
            - 45.6507 * V2.pow(2)
            - 91.4134 * V3.pow(2)
            - 76.8701 * V5.pow(2)
        )
        q = (
            -212.8531
            + 245.7998 * V1
            - 127.3395 * V2
            + 305.8461 * V3
            + 638.1605 * V4
            + 301.2118 * V5
            - 451.3796 * V6
            - 115.5485 * V7
            + 42.8351 * V1 * V2
            + 262.3775 * V1 * V3
            - 103.5274 * V1 * V4
            - 196.1568 * V1 * V5
            - 394.7975 * V1 * V6
            - 176.3341 * V1 * V7
            + 74.8291 * V2 * V3
            + 4.1557 * V2 * V4
            - 133.8683 * V2 * V5
            + 65.8711 * V2 * V6
            - 42.6911 * V2 * V7
            - 323.9363 * V3 * V4
            - 107.3983 * V3 * V5
            - 323.2353 * V3 * V6
            + 46.9172 * V3 * V7
            - 144.4199 * V4 * V5
            + 272.3729 * V4 * V6
            + 49.0799 * V4 * V7
            + 318.4706 * V5 * V6
            - 236.2498 * V5 * V7
            + 252.4848 * V6 * V7
            - 286.0182 * V4.pow(2)
            + 393.5992 * V6.pow(2)
        )
        sigma = (
            7.7696
            + 15.4344 * V1
            - 10.6190 * V2
            - 17.9367 * V3
            + 17.1385 * V4
            + 2.5026 * V5
            - 24.3010 * V6
            + 10.6058 * V7
            - 1.2041 * V1 * V2
            - 37.2207 * V1 * V3
            - 3.2265 * V1 * V4
            + 7.3121 * V1 * V5
            + 52.3994 * V1 * V6
            + 9.7485 * V1 * V7
            - 15.9371 * V2 * V3
            - 1.1706 * V2 * V4
            - 2.6297 * V2 * V5
            + 7.0225 * V2 * V6
            - 1.4938 * V2 * V7
            + 30.2786 * V3 * V4
            + 14.5061 * V3 * V5
            + 48.5021 * V3 * V6
            - 11.4857 * V3 * V7
            - 3.1381 * V4 * V5
            - 14.9747 * V4 * V6
            + 4.5204 * V4 * V7
            - 17.6907 * V5 * V6
            - 19.2489 * V5 * V7
            - 9.8219 * V6 * V7
            - 18.7356 * V1.pow(2)
            + 12.1928 * V2.pow(2)
            - 17.5460 * V4.pow(2)
            + 5.4997 * V5.pow(2)
            - 26.2718 * V6.pow(2)
        )
        return -torch.cat([wca, q, sigma], dim=-1)


class OilSorbentMixed(DiscreteTestProblem, MultiObjectiveTestProblem):
    """All objectives should be minimized.

    The reference point comes from using the infer_reference_point
    method on the full discrete search space where the continuous parameters are discretized into 100 values.
    """

    # _max_hv = 1177461.0 # full discrete case
    # _max_hv = 1279774.75 # approximate for continuous
    _discrete_values = {
        # "V1": [3 / 7, 4 / 6, 1, 6 / 4, 7 / 3],
        "V2": [0.7, 1, 1.4, 1.7, 2],
        "V3": [12, 15, 18, 21, 24],
        "V4": [0.12, 0.135, 0.15, 0.165, 0.18],
        # "V5": [0, 0.04, 0.08, 0.10, 0.20],
        "V6": [16, 20, 26, 28],
        "V7": [0.41, 0.6, 0.84, 1.32],
    }

    _bounds = [
        (0, 1),  # continuous
        (0, 4),  # 5 ordinal values
        (0, 4),  # 5 ordinal values
        (0, 4),  # 5 ordinal values
        (0, 1),  # continuous
        (0, 3),  # 4 ordinal values
        (0, 3),  # 4 ordinal values
    ]
    dim = 7
    num_objectives = 3
    # _ref_point = [-133.9736, -4.8289, 38.6565] # full discrete case
    _ref_point = [-125.3865, -57.8292, 43.2665]

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        integer_indices: Optional[List[int]] = None,
    ) -> None:
        if integer_indices is None:
            integer_indices = [1, 2, 3, 5, 6]
            # integer_indices = list(range(self.dim))
        MultiObjectiveTestProblem.__init__(
            self,
            noise_std=noise_std,
            negate=negate,
            #  integer_indices=integer_indices
        )
        self._setup(integer_indices=integer_indices)
        self.discrete_values = BufferDict()
        for k, v in self._discrete_values.items():
            self.discrete_values[k] = torch.tensor(v, dtype=torch.float)
            self.discrete_values[k] /= self.discrete_values[k].max()

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_split = list(torch.split(X, 1, -1))
        # remap from integer space to proper space
        for i, V_i in enumerate(X_split):
            name = f"V{i+1}"
            if name in self.discrete_values:

                X_split[i] = self.discrete_values[name][V_i.view(-1).long()].view(
                    V_i.shape
                )
        V1, V2, V3, V4, V5, V6, V7 = X_split
        wca = (
            -197.0928
            - 78.3309 * V1
            + 98.6355 * V2
            + 300.0701 * V3
            + 89.8360 * V4
            + 208.2343 * V5
            + 332.9341 * V6
            + 135.6621 * V7
            - 11.0715 * V1 * V2
            + 201.8934 * V1 * V3
            + 17.1270 * V1 * V4
            + 2.5198 * V1 * V5
            - 109.3922 * V1 * V6
            + 30.1607 * V1 * V7
            - 46.1790 * V2 * V3
            + 19.2888 * V2 * V4
            - 102.9493 * V2 * V5
            - 19.1245 * V2 * V6
            + 53.6297 * V2 * V7
            - 73.0649 * V3 * V4
            - 37.7181 * V3 * V5
            - 219.1268 * V3 * V6
            - 55.3704 * V3 * V7
            + 3.8778 * V4 * V5
            - 6.9252 * V4 * V6
            - 105.1650 * V4 * V7
            - 34.3181 * V5 * V6
            - 36.3892 * V5 * V7
            - 82.3222 * V6 * V7
            - 16.7536 * V1.pow(2)
            - 45.6507 * V2.pow(2)
            - 91.4134 * V3.pow(2)
            - 76.8701 * V5.pow(2)
        )
        q = (
            -212.8531
            + 245.7998 * V1
            - 127.3395 * V2
            + 305.8461 * V3
            + 638.1605 * V4
            + 301.2118 * V5
            - 451.3796 * V6
            - 115.5485 * V7
            + 42.8351 * V1 * V2
            + 262.3775 * V1 * V3
            - 103.5274 * V1 * V4
            - 196.1568 * V1 * V5
            - 394.7975 * V1 * V6
            - 176.3341 * V1 * V7
            + 74.8291 * V2 * V3
            + 4.1557 * V2 * V4
            - 133.8683 * V2 * V5
            + 65.8711 * V2 * V6
            - 42.6911 * V2 * V7
            - 323.9363 * V3 * V4
            - 107.3983 * V3 * V5
            - 323.2353 * V3 * V6
            + 46.9172 * V3 * V7
            - 144.4199 * V4 * V5
            + 272.3729 * V4 * V6
            + 49.0799 * V4 * V7
            + 318.4706 * V5 * V6
            - 236.2498 * V5 * V7
            + 252.4848 * V6 * V7
            - 286.0182 * V4.pow(2)
            + 393.5992 * V6.pow(2)
        )
        sigma = (
            7.7696
            + 15.4344 * V1
            - 10.6190 * V2
            - 17.9367 * V3
            + 17.1385 * V4
            + 2.5026 * V5
            - 24.3010 * V6
            + 10.6058 * V7
            - 1.2041 * V1 * V2
            - 37.2207 * V1 * V3
            - 3.2265 * V1 * V4
            + 7.3121 * V1 * V5
            + 52.3994 * V1 * V6
            + 9.7485 * V1 * V7
            - 15.9371 * V2 * V3
            - 1.1706 * V2 * V4
            - 2.6297 * V2 * V5
            + 7.0225 * V2 * V6
            - 1.4938 * V2 * V7
            + 30.2786 * V3 * V4
            + 14.5061 * V3 * V5
            + 48.5021 * V3 * V6
            - 11.4857 * V3 * V7
            - 3.1381 * V4 * V5
            - 14.9747 * V4 * V6
            + 4.5204 * V4 * V7
            - 17.6907 * V5 * V6
            - 19.2489 * V5 * V7
            - 9.8219 * V6 * V7
            - 18.7356 * V1.pow(2)
            + 12.1928 * V2.pow(2)
            - 17.5460 * V4.pow(2)
            + 5.4997 * V5.pow(2)
            - 26.2718 * V6.pow(2)
        )
        return -torch.cat([wca, q, sigma], dim=-1)
