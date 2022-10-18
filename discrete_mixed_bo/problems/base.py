#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from botorch.test_functions.base import BaseTestProblem, MultiObjectiveTestProblem
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor
from torch.nn import Module


class DiscreteTestProblem(BaseTestProblem):
    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        integer_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__(negate=negate, noise_std=noise_std)
        self._setup(
            integer_indices=integer_indices, categorical_indices=categorical_indices
        )

    def _setup(
        self,
        integer_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> None:
        dim = self.bounds.shape[-1]
        discrete_indices = []
        if integer_indices is None:
            integer_indices = []
        if categorical_indices is None:
            categorical_indices = []
        self.register_buffer(
            "_orig_integer_indices", torch.tensor(integer_indices, dtype=torch.long)
        )
        discrete_indices.extend(integer_indices)
        self.register_buffer(
            "_orig_categorical_indices",
            torch.tensor(sorted(categorical_indices), dtype=torch.long),
        )
        discrete_indices.extend(categorical_indices)
        if len(discrete_indices) == 0:
            raise ValueError("Expected at least one discrete feature.")
        cont_indices = sorted(list(set(range(dim)) - set(discrete_indices)))
        self.register_buffer(
            "_orig_cont_indices",
            torch.tensor(
                cont_indices,
                dtype=torch.long,
                device=self.bounds.device,
            ),
        )
        self.register_buffer("_orig_bounds", self.bounds.clone())
        # remap inputs so that categorical features come after all of
        # the ordinal features
        remapper = torch.zeros(
            self.bounds.shape[-1], dtype=torch.long, device=self.bounds.device
        )
        reverse_mapper = remapper.clone()
        for i, orig_idx in enumerate(
            cont_indices + integer_indices + categorical_indices
        ):
            remapper[i] = orig_idx
            reverse_mapper[orig_idx] = i
        self.register_buffer("_remapper", remapper)
        self.register_buffer("_reverse_mapper", reverse_mapper)
        self.bounds = self.bounds[:, remapper]
        self.register_buffer("cont_indices", reverse_mapper[cont_indices])
        self.register_buffer("integer_indices", reverse_mapper[integer_indices])
        self.register_buffer("categorical_indices", reverse_mapper[categorical_indices])

        self.effective_dim = (
            self.cont_indices.shape[0]
            + self.integer_indices.shape[0]
            + int(sum(self.categorical_features.values()))
        )

        one_hot_bounds = torch.zeros(
            2, self.effective_dim, dtype=self.bounds.dtype, device=self.bounds.device
        )
        one_hot_bounds[1] = 1
        one_hot_bounds[:, self.integer_indices] = self.integer_bounds
        one_hot_bounds[:, self.cont_indices] = self.cont_bounds
        self.register_buffer("one_hot_bounds", one_hot_bounds)

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the function on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        # remap to original space
        X = X[..., self._reverse_mapper]
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    def evaluate_slack(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the constraint function on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape x n_constraints`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        # remap to original space
        X = X[..., self._reverse_mapper]
        f = self.evaluate_slack_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        return f if batch else f.squeeze(0)

    @property
    def integer_bounds(self) -> Optional[Tensor]:
        if self.integer_indices is not None:
            return self.bounds[:, self.integer_indices]
        return None

    @property
    def cont_bounds(self) -> Optional[Tensor]:
        if self.cont_indices is not None:
            return self.bounds[:, self.cont_indices]
        return None

    @property
    def categorical_bounds(self) -> Optional[Tensor]:
        if self.categorical_indices is not None:
            return self.bounds[:, self.categorical_indices]
        return None

    @property
    def categorical_features(self) -> Optional[Dict[int, int]]:
        # Return dictionary mapping indices to cardinalities
        if self.categorical_indices is not None:
            categ_bounds = self.categorical_bounds
            return OrderedDict(
                zip(
                    self.categorical_indices.tolist(),
                    (categ_bounds[1] - categ_bounds[0] + 1).long().tolist(),
                )
            )
        return None

    @property
    def objective_weights(self) -> Optional[Tensor]:
        return None

    @property
    def is_moo(self) -> bool:
        return isinstance(self, MultiObjectiveTestProblem) and (
            self.objective_weights is None
        )


class DiscretizedBotorchTestProblem(DiscreteTestProblem):
    r"""Class for converting continuous botorch test problems into
    discrete or mixed problems.
    """

    def __init__(
        self,
        problem: BaseTestProblem,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_indices: Optional[List[int]] = None,
        categorical_bounds: Optional[Tensor] = None,
    ) -> None:
        Module.__init__(self)
        self.problem = problem
        self.dim = problem.dim
        self.register_buffer("bounds", problem.bounds.clone())
        if integer_indices is not None:
            self.bounds[:, integer_indices] = integer_bounds
        if categorical_indices is not None:
            if (categorical_bounds[1] - categorical_bounds[0] < 2).any():
                raise ValueError(
                    "Binary categorical parameters should be specified as integer parameters, not categoricals."
                )
            self.bounds[:, categorical_indices] = categorical_bounds

        self._setup(
            integer_indices=integer_indices, categorical_indices=categorical_indices
        )

    def get_orig_X(self, X: Tensor) -> Tensor:
        # normalize from integer bounds to unit cube
        unit_X = normalize(X, self._orig_bounds)
        # unnormalize from unit cube to original problem bounds
        return unnormalize(unit_X, self.problem.bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        This assumes that only discrete X are passed.
        """
        orig_X = self.get_orig_X(X=X)
        return self.problem.evaluate_true(orig_X)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        r"""
        This assumes that only discrete X are passed.
        """
        orig_X = self.get_orig_X(X=X)
        return self.problem.evaluate_slack_true(X)

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        # remap to original indices
        X = X[..., self._reverse_mapper]
        # remap to original bounds
        orig_X = self.get_orig_X(X=X)
        return self.problem.forward(X=orig_X, noise=noise)

    @property
    def __class__(self):
        return self.problem.__class__

    @property
    def ref_point(self) -> Tensor:
        return self.problem.ref_point

    @property
    def num_objectives(self) -> int:
        return self.problem.num_objectives

    @property
    def ref_num_constraints(self) -> int:
        return self.problem.num_constraints
