#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Coverage and Capacity optimization for Cell networks.

Code from: https://github.com/Ryandry1st/CCO-in-ORAN/tree/main/cco_standalone_icassp_2021

Paper: R. M. Dreifuerst, et al. Optimizing Coverage and Capacity in Cellular Networks using Machine Learning. IEEE ICASSP special session on Machine Learning in Networks, 2021.
"""
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem
from discrete_mixed_bo.problems.cco.problem_formulation import (
    CCORasterBlanketFormulation,
)
from discrete_mixed_bo.problems.cco.simulated_rsrp import SimulatedRSRP


class CCO(DiscreteTestProblem, MultiObjectiveTestProblem):
    dim: int = 30
    _ref_point = [0.35, 0.35]

    def __init__(
        self,
        data: Optional[Dict[int, Any]] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
        scalarize: bool = False,
        n_int_values: int = 6,
    ) -> None:
        """
        This method requires a `data` object that is constructed as follows:
        ```
        data = {}
        for i in range(11):
           data[i] = dict(np.load(f"powermaps/powermatrixDT{i}.npz"))
        ```
        The npz files can be retrieved from:
        https://github.com/Ryandry1st/CCO-in-ORAN/tree/main/cco_standalone_icassp_2021/data/power_maps
        """
        if n_int_values not in (6, 11):
            raise ValueError("Only 6 and 11 int values are supported")
        self._n_int_values = n_int_values
        self._bounds = [
            (0.0, n_int_values - 1) for _ in range(15)
        ] + [  # downtilts (integers)
            (30.0, 50.0) for _ in range(15)  # transmission power (floats)
        ]
        MultiObjectiveTestProblem.__init__(
            self,
            negate=negate,
            noise_std=noise_std,
        )
        self._setup(integer_indices=list(range(15)))
        if data is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "..", "data", "powermaps")
            with open(path, "rb") as f:
                data = torch.load(f)
        self.simulated_rsrp = SimulatedRSRP(
            powermaps=data,
            min_TX_power_dBm=30,
            max_TX_power_dBm=50,
        )
        self.problem_formulation = CCORasterBlanketFormulation(lambda_weight=0.9)

        _, num_sectors = self.simulated_rsrp.get_configuration_shape()
        downtilts_choices, (
            min_Tx_power_dBm,
            max_Tx_power_dBm,
        ) = self.simulated_rsrp.get_configuration_range()
        xy_min, xy_max = self.simulated_rsrp.get_locations_range()
        self.scalarize = scalarize
        self.register_buffer("_objective_weights", torch.tensor([0.5, 0.5]))

    def _powermap_evaluation_fn(self, input: Tensor) -> Tensor:
        (
            rsrp_powermap,
            interference_powermap,
            _,
        ) = self.simulated_rsrp.get_RSRP_and_interference_powermap(input.numpy())

        # Compute aggregate metrics from the powermap
        # compute percentages, we want to minimize both of these
        (
            f_weak_coverage_pct,
            g_over_coverage_pct,
        ) = self.problem_formulation.get_weak_over_coverage_area_percentages(
            rsrp_powermap, interference_powermap
        )
        return torch.tensor(
            [f_weak_coverage_pct, g_over_coverage_pct],
            dtype=input.dtype,
            device=input.device,
        )

    def evaluate_true(self, X: Tensor) -> Tensor:
        X = X.clone()
        if self._n_int_values == 6:
            X[..., :15] *= 2
        Y = (
            torch.stack(
                [
                    self._powermap_evaluation_fn(x)
                    for x in X.view(-1, 2, self.dim // 2).cpu()
                ],
            )
            .view(*X.shape[:-1], 2)
            .to(X)
        )
        if self.scalarize:
            return Y @ self._objective_weights
        else:
            return Y

    @property
    def objective_weights(self) -> Optional[Tensor]:
        # if self.scalarize:
        #     return self._objective_weights
        return None

    @property
    def is_moo(self) -> bool:
        return not self.scalarize
