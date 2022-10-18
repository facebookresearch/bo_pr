#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import numpy as np
import torch
from botorch.models.kernels import CategoricalKernel
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from torch import Tensor


def get_kernel(
    kernel_type: str,
    dim: int,
    binary_dims: List[int],
    categorical_transformed_features: Dict[int, int],
    train_X: Tensor,
    train_Y: Tensor,
    function_name: Optional[str] = None,
    use_ard_binary: bool = False,
) -> Optional[Kernel]:
    """Helper function for kernel construction."""
    # ard kernel for continuous features
    if kernel_type == "mixed_categorical":
        categorical_dims = list(categorical_transformed_features.keys())
    else:
        if len(categorical_transformed_features) > 0:
            start = min(categorical_transformed_features.keys())
            categorical_dims = list(range(start, dim))
        else:
            categorical_dims = []

    if "mixed" in kernel_type:
        cont_dims = list(set(list(range(dim))) - set(binary_dims))
        if ("latent" in kernel_type) or ("categorical" in kernel_type):
            cont_dims = list(set(cont_dims) - set(categorical_dims))
        kernels = []
        # ard kernel for continuous features
        if len(cont_dims) > 0:
            kernels.append(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=len(cont_dims),
                    active_dims=cont_dims,
                    lengthscale_constraint=Interval(0.1, 20.0),
                )
            )
        # isotropic kernel for binary features
        if len(binary_dims) > 0:
            kernels.append(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=len(binary_dims) if use_ard_binary else None,
                    active_dims=binary_dims,
                    lengthscale_constraint=Interval(0.1, 20.0),
                )
            )
        if kernel_type == "mixed_categorical":
            if len(categorical_dims) > 0:
                kernels.append(
                    CategoricalKernel(
                        ard_num_dims=len(categorical_dims),
                        active_dims=categorical_dims,
                        lengthscale_constraint=Interval(1e-3, 20.0),
                    )
                )
        elif kernel_type == "mixed_latent":
            for start, latent_dim in categorical_transformed_features.items():
                kernels.append(
                    MaternKernel(
                        # Use a isotropic kernel --
                        # one kernel for each set of latent embeddings
                        ard_num_dims=None,
                        active_dims=list(range(start, start + latent_dim)),
                        lengthscale_constraint=Interval(1e-3, 20.0),
                    )
                )

        prod_kernel = kernels[0]
        for k in kernels[1:]:
            prod_kernel *= k
        if kernel_type != "mixed_categorical":
            return ScaleKernel(prod_kernel)
        sum_kernel = kernels[0]
        for k in kernels[1:]:
            prod_kernel *= k
            sum_kernel += k
        return ScaleKernel(prod_kernel) + ScaleKernel(sum_kernel)
    elif kernel_type == "botorch_default":
        return None
    elif kernel_type == "ard_combo":
        return CombinatorialCovarModule(ard_num_dims=dim)
    elif kernel_type == "iso_combo":
        return CombinatorialCovarModule(ard_num_dims=None)
    raise ValueError(f"{kernel_type} is not supported.")


class CombinatorialCovarModule(Kernel):
    r"""This kernel is suitable for a {0, 1}^d domain and used for combo design."""

    def __init__(self, ard_num_dims=None, **kwargs):
        super().__init__(**kwargs)
        use_ard = ard_num_dims is not None and ard_num_dims > 1
        lengthscale_constraint = Interval(0.1, 20.0) if use_ard else None
        lengthscale_prior = GammaPrior(2.0, 1.0) if use_ard else None
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            lengthscale_constraint=lengthscale_constraint,
            lengthscale_prior=lengthscale_prior,
            active_dims=kwargs.get("active_dims"),
        )
        covar_module = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_prior=LogNormalPrior(9.0, 3.0),  # Flexible prior
            outputscale_constraint=Interval(1e-3, 1e3),
        )
        self._covar_module = covar_module

    def forward(self, x1, x2, diag=False, **params):
        return self._covar_module.forward(x1, x2, diag=diag, **params)
