#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A wrapper classes around AquisitionFunctions to modify inputs and outputs.
"""

from __future__ import annotations

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module


class AcquisitionFunctionWrapper(AcquisitionFunction):
    r"""Abstract acquisition wrapper."""

    def __init__(self, acq_function: AcquisitionFunction) -> None:
        Module.__init__(self)
        self.acq_function = acq_function

    @property
    def X_baseline(self) -> Optional[Tensor]:
        return self.acq_function.X_baseline

    @property
    def model(self) -> Model:
        return self.acq_function.model


class IntegratedAcquisitionFunction(AcquisitionFunction):
    r"""Integrate acquisition function wrapper.

    This can be used for integrating over batch dimensions. For example,
    this can be used to integrate over hyperparameter samples from a
    fully bayesian (MCMC) model.
    """

    def __init__(
        self, acq_function: AcquisitionFunction, marginalize_dim: int = -1
    ) -> None:
        super().__init__(acq_function=acq_function)
        self._marginalize_dim = marginalize_dim

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function and integrate over marginalize_dim.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of acquisition function values at the given
            design points `X`.
        """
        return self.acq_function(X=X).mean(dim=self._marginalize_dim)
