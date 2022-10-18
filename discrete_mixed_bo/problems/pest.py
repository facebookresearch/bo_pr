#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Problems with only binary variables.
"""
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x, seed=None):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(
            init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
        )
    else:
        init_pest_frac = np.random.beta(
            init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
        )
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(
                spread_alpha, spread_beta, size=(n_simulations,)
            )
        else:
            spread_rate = np.random.beta(
                spread_alpha, spread_beta, size=(n_simulations,)
            )
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(
                    control_alpha, control_beta[x[i]], size=(n_simulations,)
                )
            else:
                control_rate = np.random.beta(
                    control_alpha, control_beta[x[i]], size=(n_simulations,)
                )
            next_pest_frac = _pest_spread(
                curr_pest_frac, spread_rate, control_rate, True
            )
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                1.0
                - control_price_max_discount[x[i]]
                / float(n_stages)
                * float(np.sum(x == x[i]))
            )
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


class PestControl(DiscreteTestProblem):
    """
    Pest Control Problem.
    """

    def __init__(
        self,
        dim: int = 25,
        n_choice: int = 5,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self._bounds = [(0, n_choice - 1) for _ in range(dim)]
        self.dim = dim
        super().__init__(
            negate=negate,
            noise_std=noise_std,
            categorical_indices=list(range(self.dim)),
        )
        self.seed = 0

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.compute(X.cpu().view(-1, self.dim)).to(X).view(X.shape[:-1])

    def compute(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.int()
        if x.dim() == 1:
            x = x.reshape(1, -1)
        res = torch.tensor([self._compute(x_) for x_ in x])
        # # Add a small ammount of noise to prevent training instabilities
        # res += 1e-6 * torch.randn_like(res)
        return res

    def _compute(self, x):
        assert x.numel() == self.dim
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = _pest_control_score(
            (x.cpu() if x.is_cuda else x).numpy(), seed=self.seed
        )
        # evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy(), seed=None)
        res = float(evaluation) * x.new_ones((1,)).float()
        return res
