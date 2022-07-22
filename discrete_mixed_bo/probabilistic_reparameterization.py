#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Probabilistic Reparameterization (with gradients) using Monte Carlo estimators.
"""

from abc import ABC, abstractmethod
import torch
from contextlib import ExitStack
from collections import OrderedDict
from torch.nn import Module
from torch import Tensor
from torch.autograd import Function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
)
from discrete_mixed_bo.wrapper import AcquisitionFunctionWrapper
from typing import Dict, List, Optional
from discrete_mixed_bo.input import (
    AnalyticProbabilisticReparameterizationInputTransform,
    MCProbabilisticReparameterizationInputTransform,
    OneHotToNumeric,
)


def get_probabilistic_reparameterization_input_transform(
    dim: int,
    integer_indices: List[int],
    integer_bounds: Tensor,
    categorical_features: Optional[Dict[int, int]] = None,
    use_analytic: bool = False,
    mc_samples: int = 1024,
    resample: bool = False,
    flip: bool = False,
    tau: float = 0.1,
) -> ChainedInputTransform:
    bounds = torch.zeros(
        2, dim, dtype=integer_bounds.dtype, device=integer_bounds.device
    )
    bounds[1] = 1
    bounds[:, integer_indices] = integer_bounds
    tfs = OrderedDict()
    if integer_indices is not None and len(integer_indices) > 0:
        # unnormalize to integer space
        tfs["unnormalize"] = Normalize(
            d=dim,
            bounds=bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=True,
        )
        # round
    if use_analytic:
        tfs["round"] = AnalyticProbabilisticReparameterizationInputTransform(
            dim=dim,
            integer_indices=integer_indices,
            integer_bounds=integer_bounds,
            categorical_features=categorical_features,
            tau=tau,
        )
    else:
        tfs["round"] = MCProbabilisticReparameterizationInputTransform(
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            resample=resample,
            mc_samples=mc_samples,
            flip=flip,
            tau=tau,
        )
    if integer_indices is not None and len(integer_indices) > 0:
        # normalize to unit cube
        tfs["normalize"] = Normalize(
            d=dim,
            bounds=bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=False,
        )
    tf = ChainedInputTransform(**tfs)
    tf.eval()
    return tf


class AbstractProbabilisticReparameterization(AcquisitionFunctionWrapper, ABC):
    """Acquisition Function Wrapper that leverages probabilistic reparameterization."""

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        dim: int,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        batch_limit: Optional[int] = None,
        apply_numeric: bool = False,
        **kwargs
    ) -> None:
        if categorical_features is None and (
            integer_indices is None or integer_bounds is None
        ):
            raise NotImplementedError(
                "categorical_features or integer indices and integer_bounds must be provided."
            )
        super().__init__(acq_function=acq_function)
        self.batch_limit = batch_limit

        if apply_numeric:
            self.one_hot_to_numeric = OneHotToNumeric(
                categorical_features=categorical_features,
                transform_on_train=False,
                transform_on_eval=True,
                transform_on_fantasize=False,
            )
            self.one_hot_to_numeric.eval()
        else:
            self.one_hot_to_numeric = None
        discrete_indices = []
        if integer_indices is not None:
            self.register_buffer(
                "integer_indices",
                torch.tensor(
                    integer_indices, dtype=torch.long, device=integer_bounds.device
                ),
            )
            discrete_indices.extend(integer_indices)
        else:
            self.register_buffer(
                "integer_indices",
                torch.tensor([], dtype=torch.long, device=integer_bounds.device),
            )
        if categorical_features is not None and len(categorical_features) > 0:
            categorical_indices = list(range(min(categorical_features.keys()), dim))
            discrete_indices.extend(categorical_indices)
            self.register_buffer(
                "categorical_indices",
                torch.tensor(
                    categorical_indices,
                    dtype=torch.long,
                    device=integer_bounds.device,
                ),
            )
            self.categorical_features = categorical_features
        else:
            self.register_buffer(
                "categorical_indices",
                torch.tensor(
                    [],
                    dtype=torch.long,
                    device=integer_bounds.device,
                ),
            )

        self.register_buffer(
            "cont_indices",
            torch.tensor(
                sorted(list(set(range(dim)) - set(discrete_indices))),
                dtype=torch.long,
                device=integer_bounds.device,
            ),
        )
        self.model = acq_function.model  # for sample_around_best heuristic

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Compute PR."""
        pass


class AnalyticProbabilisticReparameterization(AbstractProbabilisticReparameterization):
    """Acquisition Function Wrapper that leverages analytic probabilistic reparameterization.

    Note: this is only reasonable from a computation perspective for relatively small numbers of discrete options (probably less than a few thousand).
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        batch_limit: Optional[int] = None,
        apply_numeric: bool = False,
        tau: float = 0.1,
    ) -> None:
        super().__init__(
            acq_function=acq_function,
            dim=dim,
            integer_indices=integer_indices,
            integer_bounds=integer_bounds,
            categorical_features=categorical_features,
            batch_limit=batch_limit,
            apply_numeric=apply_numeric,
        )
        # create input transform
        # need to compute cross product of discrete options and weights
        self.input_transform = get_probabilistic_reparameterization_input_transform(
            dim=dim,
            use_analytic=True,
            integer_indices=integer_indices,
            integer_bounds=integer_bounds,
            categorical_features=categorical_features,
            tau=tau,
        )
        self.input_transform.to(dtype=dtype, device=device)
        if self.batch_limit is None:
            self.batch_limit = self.input_transform["round"].all_discrete_options.shape[
                0
            ]

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate PR."""
        X_discrete_all = self.input_transform(X.unsqueeze(-3))
        acq_values_list = []
        start_idx = 0
        if self.one_hot_to_numeric is not None:
            X_discrete_all = self.one_hot_to_numeric(X_discrete_all)
        if X.shape[-2] != 1:
            raise NotImplementedError

        # save the probabilities
        if "unnormalize" in self.input_transform:
            unnormalized_X = self.input_transform["unnormalize"](X)
        else:
            unnormalized_X = X
        # this is batch_shape x n_discrete (after squeezing)
        probs = self.input_transform["round"].get_probs(X=unnormalized_X).squeeze(-1)
        # TODO: filter discrete configs with zero probability
        # this requires padding because there may be a different number in each batch. Each batch bucket needs at least
        # nonzero_prob.sum(dim=-1).max() elements to avoid ragged tensors
        # nonzero_prob = probs > 0
        # try:
        #     X_discrete_all = X_discrete_all[nonzero_prob].view(*X_discrete_all.shape[:-3], -1, *X_discrete_all.shape[-2:])
        # except RuntimeError:
        #     import pdb
        #     pdb.set_trace()
        # probs = probs[nonzero_prob].view(*probs.shape[:-1], -1)
        while start_idx < X_discrete_all.shape[-3]:
            end_idx = min(start_idx + self.batch_limit, X_discrete_all.shape[-3])
            acq_values = self.acq_function(X_discrete_all[..., start_idx:end_idx, :, :])
            acq_values_list.append(acq_values)
            start_idx += self.batch_limit
        # this is batch_shape x n_discrete
        acq_values = torch.cat(acq_values_list, dim=-1)
        # now weight the acquisition values by probabilities
        return (acq_values * probs).sum(dim=-1)


class MCProbabilisticReparameterization(AbstractProbabilisticReparameterization):
    """Acquisition Function Wrapper that leverages MC-based probabilistic reparameterization."""

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        dim: int,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        batch_limit: Optional[int] = None,
        apply_numeric: bool = False,
        mc_samples: int = 1024,
        grad_estimator: str = "reinforce",
        tau: float = 0.1,
    ) -> None:
        super().__init__(
            acq_function=acq_function,
            dim=dim,
            integer_indices=integer_indices,
            integer_bounds=integer_bounds,
            categorical_features=categorical_features,
            batch_limit=batch_limit,
            apply_numeric=apply_numeric,
        )
        if self.batch_limit is None:
            self.batch_limit = mc_samples
        self.grad_estimator = grad_estimator
        self._pr_acq_function = _MCProbabilisticReparameterization()
        # create input transform
        self.input_transform = get_probabilistic_reparameterization_input_transform(
            dim=dim,
            integer_indices=integer_indices,
            integer_bounds=integer_bounds,
            categorical_features=categorical_features,
            mc_samples=mc_samples,
            tau=tau,
        )

        if grad_estimator in ("arm", "u2g"):
            self.input_transform_flip = (
                get_mc_probabilistic_reparameterization_input_transform(
                    dim=dim,
                    integer_indices=integer_indices,
                    integer_bounds=integer_bounds,
                    categorical_features=categorical_features,
                    mc_samples=mc_samples,
                    flip=True,
                )
            )
        else:
            self.input_transform_flip = None

    def forward(self, X: Tensor) -> Tensor:
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return self._pr_acq_function.apply(
            X,
            self.acq_function,
            self.input_transform,
            self.input_transform_flip,
            self.batch_limit,
            self.integer_indices,
            self.cont_indices,
            self.categorical_indices,
            self.grad_estimator,
            self.one_hot_to_numeric,
        )


class _MCProbabilisticReparameterization(Function):
    r"""Evaluate the acquisition function use a custom MC gradient estimator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        acq_function: AcquisitionFunction,
        input_tf: InputTransform,
        input_tf_flip: Optional[InputTransform],
        batch_limit: Optional[int],
        integer_indices: Tensor,
        cont_indices: Tensor,
        categorical_indices: Tensor,
        grad_estimator: str,
        one_hot_to_numeric: Optional[OneHotToNumeric],
    ):
        """Evaluate the expectation of the acquisition function under
        probabilistic reparameterization. Compute this in chunks of size
        batch_limit to enable scaling to large numbers of samples from the
        proposal distribution.
        """
        with ExitStack() as es:
            if ctx.needs_input_grad[0]:
                es.enter_context(torch.enable_grad())
            if cont_indices.shape[0] > 0:
                # only require gradient for continuous parameters
                ctx.cont_input = input[..., cont_indices].detach().requires_grad_(True)
                cont_idx = 0
                cols = []
                for col in range(input.shape[-1]):
                    # cont_indices is sorted in ascending order
                    if (
                        cont_idx < cont_indices.shape[0]
                        and col == cont_indices[cont_idx]
                    ):
                        cols.append(ctx.cont_input[..., cont_idx])
                        cont_idx += 1
                    else:
                        cols.append(input[..., col])
                input = torch.stack(cols, dim=-1)
            else:
                ctx.cont_input = None
            ctx.input = input
            ctx.integer_indices = integer_indices
            ctx.discrete_indices = input_tf["round"].discrete_indices
            ctx.cont_indices = cont_indices
            ctx.categorical_indices = categorical_indices
            tilde_x_samples = input_tf(input.unsqueeze(-3))
            # save the rounding component

            rounding_component = tilde_x_samples.clone()
            if integer_indices.shape[0] > 0:
                input_integer_params = input[..., integer_indices].unsqueeze(-3)
                rounding_component[..., integer_indices] = (
                    (tilde_x_samples[..., integer_indices] - input_integer_params > 0)
                    | (input_integer_params == 1)
                ).to(tilde_x_samples)
            if categorical_indices.shape[0] > 0:
                rounding_component[..., categorical_indices] = tilde_x_samples[
                    ..., categorical_indices
                ]
            ctx.rounding_component = rounding_component[..., ctx.discrete_indices]
            ctx.tau = input_tf["round"].tau
            if hasattr(input_tf["round"], "base_samples"):
                ctx.base_samples = input_tf["round"].base_samples.detach()
            # save the probabilities
            if "unnormalize" in input_tf:
                unnormalized_input = input_tf["unnormalize"](input)
            else:
                unnormalized_input = input
            # this is only for the integer parameters
            ctx.prob = input_tf["round"].get_rounding_prob(unnormalized_input)

            if categorical_indices.shape[0] > 0:
                ctx.base_samples_categorical = input_tf[
                    "round"
                ].base_samples_categorical.clone()
            # compute the acquisition function where inputs are rounded according to base_samples < prob
            ctx.tilde_x_samples = tilde_x_samples
            ctx.grad_estimator = grad_estimator
            acq_values_list = []
            start_idx = 0
            if one_hot_to_numeric is not None:
                tilde_x_samples = one_hot_to_numeric(tilde_x_samples)

            while start_idx < tilde_x_samples.shape[-3]:
                end_idx = min(start_idx + batch_limit, tilde_x_samples.shape[-3])
                acq_values = acq_function(tilde_x_samples[..., start_idx:end_idx, :, :])
                acq_values_list.append(acq_values)
                start_idx += batch_limit
            acq_values = torch.cat(acq_values_list, dim=-1)
            ctx.mean_acq_values = acq_values.mean(
                dim=-1
            )  # average over samples from proposal distribution
            ctx.acq_values = acq_values
            if grad_estimator in ("arm", "u2g"):
                if input_tf["round"].categorical_starts.shape[0] > 0:
                    raise NotImplementedError
                # use the same base samples in input_tf_flip
                if (
                    not hasattr(input_tf_flip, "base_samples")
                    or input_tf_flip.base_samples.shape[-2] != input.shape[-2]
                ):
                    input_tf_flip["round"].base_samples = (
                        input_tf["round"].base_samples.detach().clone()
                    )

                ctx.base_samples = input_tf_flip["round"].base_samples.detach()
                with torch.no_grad():
                    tilde_x_samples_flip = input_tf_flip(input.unsqueeze(-3))
                    # save the rounding component
                    ctx.rounding_component_flip = (
                        tilde_x_samples_flip[..., integer_indices]
                        - input[..., integer_indices].unsqueeze(-3)
                        > 0
                    ).to(tilde_x_samples_flip)
                    # compute the acquisition function where inputs are rounded according to base_samples > 1-prob
                    # This is used for the ARM/U2G gradient estimators
                    if one_hot_to_numeric is not None:
                        tilde_x_samples_flip = one_hot_to_numeric(tilde_x_samples_flip)
                    acq_values_flip_list = []
                    start_idx = 0
                    while start_idx < tilde_x_samples_flip.shape[-3]:
                        end_idx = min(
                            start_idx + batch_limit, tilde_x_samples_flip.shape[-3]
                        )
                        acq_values_flip = acq_function(
                            tilde_x_samples_flip[..., start_idx:end_idx, :, :]
                        )
                        acq_values_flip_list.append(acq_values_flip)
                        start_idx += batch_limit
                    acq_values_flip = torch.cat(acq_values_flip_list, dim=-1)
                    ctx.mean_acq_values_flip = acq_values_flip.mean(
                        dim=-1
                    )  # average over samples from proposal distribution
                    ctx.acq_values_flip = acq_values_flip
            return ctx.mean_acq_values.detach()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient of the expectation of the acquisition function
        with respect to the parameters of the proposal distribution using
        Monte Carlo.
        """
        # this is overwriting the entire gradient w.r.t. x'
        # x' has shape batch_shape x q x d
        if ctx.needs_input_grad[0]:
            acq_values = ctx.acq_values
            mean_acq_values = ctx.mean_acq_values

            tilde_x_samples = ctx.tilde_x_samples
            integer_indices = ctx.integer_indices
            cont_indices = ctx.cont_indices
            discrete_indices = ctx.discrete_indices
            rounding_component = ctx.rounding_component
            input = ctx.input
            # retrieve only the ordinal parameters
            expanded_acq_values = acq_values.view(*acq_values.shape, 1, 1).expand(
                acq_values.shape + rounding_component.shape[-2:]
            )
            prob = ctx.prob.unsqueeze(-3)
            if ctx.grad_estimator in ("arm", "u2g"):
                rounding_component_flip = ctx.rounding_component_flip
                acq_values_flip = ctx.acq_values_flip
                mean_acq_values_flip = ctx.mean_acq_values_flip
                expanded_acq_values_flip = acq_values_flip.view(
                    *acq_values_flip.shape, 1, 1
                ).expand(acq_values_flip.shape + rounding_component_flip.shape[-2:])
                if ctx.grad_estimator == "arm":
                    sample_level = (
                        (expanded_acq_values_flip - expanded_acq_values)
                        * (ctx.base_samples - 0.5)
                        * torch.abs(rounding_component_flip - rounding_component)
                    )
                elif ctx.grad_estimator == "u2g":
                    prob_abs = prob.clone()
                    prob_abs[prob_abs < 0.5] = 1 - prob_abs[prob_abs < 0.5]
                    sample_level = (
                        0.5
                        * (expanded_acq_values_flip - expanded_acq_values)
                        * prob_abs
                        * (rounding_component_flip - rounding_component)
                    )
            elif ctx.grad_estimator == "reinforce":
                sample_level = expanded_acq_values * (rounding_component - prob)
            grads = (sample_level / ctx.tau).mean(dim=-3)

            new_grads = (
                grad_output.view(
                    *grad_output.shape,
                    *[1 for _ in range(grads.ndim - grad_output.ndim)]
                )
                .expand(*grad_output.shape, *input.shape[-2:])
                .clone()
            )
            # multiply upstream grad_output by new gradients
            new_grads[..., discrete_indices] *= grads
            # use autograd for gradients w.r.t. the continuous parameters
            if ctx.cont_input is not None:
                auto_grad = torch.autograd.grad(
                    # note: this multiplies the gradient of mean_acq_values w.r.t to input
                    # by grad_output
                    mean_acq_values,
                    ctx.cont_input,
                    grad_outputs=grad_output,
                )[0]
                # overwrite grad_output since the previous step already applied the chain rule
                new_grads[..., cont_indices] = auto_grad
            return new_grads, None, None, None, None, None, None, None, None, None
        return None, None, None, None, None, None, None, None, None, None
