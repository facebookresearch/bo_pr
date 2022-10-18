#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Utilities for random fourier features.
"""

import torch
from botorch.models import ModelListGP, MultiTaskGP
from botorch.models.converter import batched_to_model_list
from botorch.models.deterministic import DeterministicModel, GenericDeterministicModel
from botorch.models.model import Model
from botorch.utils.gp_sampling import (
    RandomFourierFeatures,
    get_deterministic_model,
    get_deterministic_model_multi_samples,
    get_weights_posterior,
)


def get_gp_samples(
    model: Model,
    num_outputs: int,
    n_samples: int,
    num_rff_features: int = 512,
) -> GenericDeterministicModel:
    r"""Sample functions from GP posterior using RFFs. The returned
    `GenericDeterministicModel` effectively wraps `num_outputs` models,
    each of which has a batch shape of `n_samples`. Refer
    `get_deterministic_model_multi_samples` for more details.

    Args:
        model: The model.
        num_outputs: The number of outputs.
        n_samples: The number of functions to be sampled IID.
        num_rff_features: The number of random Fourier features.

    Returns:
        A batched `GenericDeterministicModel` that batch evaluates `n_samples`
        sampled functions.
    """
    if num_outputs > 1:
        if not isinstance(model, ModelListGP):
            models = batched_to_model_list(model).models
        else:
            models = model.models
    else:
        models = [model]
    if isinstance(models[0], MultiTaskGP):
        raise NotImplementedError

    weights = []
    bases = []
    for m in range(num_outputs):
        train_X = models[m].train_inputs[0]
        train_targets = models[m].train_targets
        # get random fourier features
        # sample_shape controls the number of iid functions.
        basis = RandomFourierFeatures(
            kernel=models[m].covar_module,
            input_dim=train_X.shape[-1],
            num_rff_features=num_rff_features,
            sample_shape=torch.Size([n_samples] if n_samples > 1 else []),
        )
        bases.append(basis)
        # TODO: when batched kernels are supported in RandomFourierFeatures,
        # the following code can be uncommented.
        # if train_X.ndim > 2:
        #    batch_shape_train_X = train_X.shape[:-2]
        #    dataset_shape = train_X.shape[-2:]
        #    train_X = train_X.unsqueeze(-3).expand(
        #        *batch_shape_train_X, n_samples, *dataset_shape
        #    )
        #    train_targets = train_targets.unsqueeze(-2).expand(
        #        *batch_shape_train_X, n_samples, dataset_shape[0]
        #    )
        phi_X = basis(train_X)
        # Sample weights from bayesian linear model
        # 1. When inputs are not batched, train_X.shape == (n, d)
        # weights.sample().shape == (n_samples, num_rff_features)
        # 2. When inputs are batched, train_X.shape == (batch_shape_input, n, d)
        # This is expanded to (batch_shape_input, n_samples, n, d)
        # to maintain compatibility with RFF forward semantics
        # weights.sample().shape == (batch_shape_input, n_samples, num_rff_features)
        mvn = get_weights_posterior(
            X=phi_X,
            y=train_targets,
            sigma_sq=models[m].likelihood.noise.mean().item(),
        )
        weights.append(mvn.sample())

    # TODO: Ideally support RFFs for multi-outputs instead of having to
    # generate a basis for each output serially.
    if n_samples > 1:
        return get_deterministic_model_multi_samples(
            weights=weights,
            bases=bases,
        )
    return get_deterministic_model(
        weights=weights,
        bases=bases,
    )


def get_gp_sample_w_transforms(
    model: Model,
    num_outputs: int,
    n_samples: int,
    num_rff_features: int = 512,
) -> DeterministicModel:
    intf = None
    octf = None
    if hasattr(model, "input_transform"):
        intf = model.input_transform
    if hasattr(model, "outcome_transform"):
        octf = model.outcome_transform
        model.outcome_transform = None
    base_gp_samples = get_gp_samples(
        model=model,
        num_outputs=num_outputs,
        n_samples=n_samples,
        num_rff_features=num_rff_features,
    )
    if intf is not None:
        base_gp_samples.input_transform = intf
        model.input_transform = intf
    if octf is not None:
        base_gp_samples.outcome_transform = octf
        model.outcome_transform = octf
    return base_gp_samples
