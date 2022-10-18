#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
Trust Region Utilities.
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class TurboState:
    dim: int
    batch_size: int
    is_constrained: bool
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")  # Goal is maximization
    constraint_violation = float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if not state.is_constrained:
        better_than_current = Y_next.max() > state.best_value + 1e-3 * math.fabs(
            state.best_value
        )
        state.best_value = max(state.best_value, Y_next.max().item())
    else:
        feas = (Y_next[..., 1:] >= 0).all(dim=-1)
        if feas.any() and state.constraint_violation == 0:  # (1) Both are feasible
            new_best = Y_next[feas, 0].max()
            better_than_current = new_best > state.best_value + 1e-3 * math.fabs(
                state.best_value
            )
            state.best_value = max(state.best_value, Y_next[feas, 0].max().item())
        elif feas.any() and state.constraint_violation > 0:  # (2) New is feasible
            better_than_current = True
            state.best_value = Y_next[feas, 0].max().item()
            state.constraint_violation = 0.0
        elif not feas.any() and state.constraint_violation > 0:  # (3) None are feasible
            violation = torch.clamp_max(Y_next[..., 1:], 0.0).abs().sum(dim=-1)
            better_than_current = (
                violation.min()
                < state.constraint_violation
                - 1e-3 * math.fabs(state.constraint_violation)
            )
            state.constraint_violation = min(
                state.constraint_violation, violation.min().item()
            )
        else:  # All of these count as failures
            better_than_current = False

    if better_than_current:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
    return state
