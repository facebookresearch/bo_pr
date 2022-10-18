#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, Optional, Tuple

import gpytorch.settings as gpt_settings
import numpy as np
import torch
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from torch import Tensor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class Chemistry(DiscreteTestProblem):
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        if data is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "data", "chem_model_info")
            with open(path, "rb") as f:
                data = torch.load(f)

        self.dim = 5
        bounds = data["bounds"]
        self._bounds = bounds.t().tolist()
        super().__init__(
            negate=negate, noise_std=noise_std, categorical_indices=list(range(3))
        )
        self.register_buffer("_model_bounds", bounds)
        # construct surrogate
        X_norm = normalize(data["X"], bounds=self._model_bounds)
        Y = data["Y"]
        train_Yvar = torch.full_like(Y, 1e-10) * Y.std().pow(2)
        # override the default min fixed noise level
        # this requires https://github.com/cornellius-gp/gpytorch/pull/2132
        lb = float("-inf")
        with gpt_settings.min_fixed_noise(
            float_value=lb, double_value=lb, half_value=lb
        ):
            self.model = MixedSingleTaskGP(
                train_X=X_norm,
                train_Y=Y,
                cat_dims=list(range(3)),
                outcome_transform=Standardize(m=1),
                likelihood=FixedNoiseGaussianLikelihood(noise=train_Yvar.squeeze(-1)),
            )
        self.model.load_state_dict(data["state_dict"])

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_norm = normalize(X, self._model_bounds)
        with torch.no_grad():
            return -self.model.posterior(X_norm.unsqueeze(-2)).mean.view(
                X_norm.shape[:-1]
            )
