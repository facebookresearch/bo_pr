#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from botorch.models.transforms.input import InputTransform
from botorch.utils.sampling import (
    draw_sobol_normal_samples,
    draw_sobol_samples,
    sample_simplex,
)
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.module import Module as GPyTorchModule
from gpytorch.priors import NormalPrior
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import Module
from torch.nn.functional import one_hot

from discrete_mixed_bo.ste import OneHotArgmaxSTE, OneHotToNumericSTE, RoundSTE


class OneHotToNumeric(InputTransform, Module):
    r"""Transformation that maps categorical parameters from one-hot representation to numeric representation.

    This assumes that the categoricals are the trailing dimensions
    """

    def __init__(
        self,
        categorical_features: Optional[Dict[int, int]] = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
        use_ste: bool = False,
    ) -> None:
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.categorical_starts = []
        self.categorical_ends = []
        self.categorical_features = (
            None
            if ((categorical_features is None) or (len(categorical_features) == 0))
            else categorical_features
        )

        if self.categorical_features is not None:
            start_idx = None
            for i in sorted(categorical_features.keys()):
                if start_idx is None:
                    start_idx = i

                self.categorical_starts.append(start_idx)
                end_idx = start_idx + categorical_features[i]
                self.categorical_ends.append(end_idx)
                start_idx = end_idx
            self.numeric_dim = min(self.categorical_starts) + len(categorical_features)
        self.use_ste = use_ste

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of rounded inputs.
        """
        if self.categorical_features is not None:
            X_numeric = X[..., : self.numeric_dim].clone()
            idx = self.categorical_starts[0]
            for start, end in zip(self.categorical_starts, self.categorical_ends):
                if self.use_ste:
                    X_numeric[..., idx] = OneHotToNumericSTE.apply(X[..., start:end])
                else:
                    X_numeric[..., idx] = X[..., start:end].argmax(dim=-1)
                idx += 1
            return X_numeric
        return X

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs to a model.

        Un-transforms of the individual transforms are applied in reverse sequence.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        if X.requires_grad:
            raise NotImplementedError
        if self.categorical_features is not None:
            one_hot_categoricals = [
                one_hot(X[..., idx].long(), num_classes=cardinality)
                for idx, cardinality in sorted(
                    self.categorical_features.items(), key=lambda x: x[0]
                )
            ]
            X = torch.cat(
                [
                    X[..., : min(self.categorical_features.keys())],
                    *one_hot_categoricals,
                ],
                dim=-1,
            )
        return X


class Round(InputTransform, Module):
    r"""A rounding transformation for integer inputs.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization). 1. These are unnormalized back to the raw input space.
    2. The integers are rounded. 3. All values are normalized to the unit
    cube.

    In train() mode, the inputs can either (a) be normalized to the unit
    cube or (b) provided using their raw values. In the case of (a)
    transform_on_train should be set to True, so that the normalized inputs
    are unnormalized before rounding. In the case of (b) transform_on_train
    should be set to False, so that the raw inputs are rounded and then
    normalized to the unit cube.

    This transformation uses differentiable approximate rounding by default.
    The rounding function is approximated with a piece-wise function where
    each piece is a hyperbolic tangent function.

    Example:
        >>> unnormalize_tf = Normalize(
        >>>     d=d,
        >>>     bounds=bounds,
        >>>     transform_on_eval=True,
        >>>     transform_on_train=True,
        >>>     reverse=True,
        >>> )
        >>> round_tf = Round(integer_indices)
        >>> normalize_tf = Normalize(d=d, bounds=bounds)
        >>> tf = ChainedInputTransform(
        >>>     tf1=unnormalize_tf, tf2=round_tf, tf3=normalize_tf
        >>> )
    """

    def __init__(
        self,
        integer_indices: Optional[List[int]] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        approximate: bool = True,
        tau: float = 1e-3,
        use_ste: bool = False,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the integer inputs.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            approximate: A boolean indicating whether approximate or exact
                rounding should be used. Default: approximate.
            tau: The temperature parameter for approximate rounding.
            use_ste: use straight-through gradient estimator
        """
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        integer_indices = integer_indices or []
        self.register_buffer(
            "integer_indices", torch.tensor(integer_indices, dtype=torch.long)
        )
        self.categorical_starts = []
        self.categorical_ends = []
        if categorical_features is not None:
            start_idx = None
            for i in sorted(categorical_features.keys()):
                if start_idx is None:
                    start_idx = i

                self.categorical_starts.append(start_idx)
                end_idx = start_idx + categorical_features[i]
                self.categorical_ends.append(end_idx)
                start_idx = end_idx
        self.approximate = approximate
        self.tau = tau
        self.use_ste = use_ste

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of rounded inputs.
        """
        X_rounded = X.clone()
        # round integers
        X_int = X_rounded[..., self.integer_indices]
        if self.approximate:
            X_int = approximate_round(X_int, tau=self.tau)
        elif self.use_ste:
            X_int = RoundSTE.apply(X_int)
        else:
            X_int = X_int.round()
        X_rounded[..., self.integer_indices] = X_int
        # discrete categoricals to the category with the largest value
        # in the continuous relaxation of the one-hot encoding
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            cardinality = end - start
            if self.approximate:
                raise NotImplementedError
            elif self.use_ste:
                X_rounded[..., start:end] = OneHotArgmaxSTE.apply(
                    X[..., start:end],
                    cardinality,
                )
            else:
                X_rounded[..., start:end] = one_hot(
                    X[..., start:end].argmax(dim=-1), num_classes=cardinality
                )
        return X_rounded


class AnalyticProbabilisticReparameterizationInputTransform(InputTransform, Module):
    r"""Probabilistic reparameterization input transform.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization).
    1. These are unnormalized back to the raw input space.
    2. The discrete values are created.
    3. All values are normalized to the unitcube.

    Example:
        >>> unnormalize_tf = Normalize(
        >>>     d=d,
        >>>     bounds=bounds,
        >>>     transform_on_eval=True,
        >>>     transform_on_train=True,
        >>>     reverse=True,
        >>> )
        >>> pr = ProbabilisticReparameterizationInputTransform(integer_indices)
        >>> normalize_tf = Normalize(d=d, bounds=bounds)
        >>> tf = ChainedInputTransform(
        >>>     tf1=unnormalize_tf, tf2=pr, tf3=normalize_tf
        >>> )
    """

    def __init__(
        self,
        dim: int,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        tau: float = 0.1,
    ) -> None:
        r"""Initialize transform.

        Args:
            integer_indices: The indices of the integer inputs.
            categorical_features: The indices and cardinality of
                each categorical feature. The features are assumed
                to be one-hot encoded. TODO: generalize to support
                alternative representations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            mc_samples: The number of MC samples.
            resample: A boolean indicating whether to resample base samples
                at each forward pass.
            flip: A boolean indicating whether round based on u < p or 1 - p < u.
            tau: The temperature parameter.
        """
        super().__init__()
        if integer_indices is None and categorical_features is None:
            raise ValueError(
                "integer_indices and/or categorical_features must be provided."
            )
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        discrete_indices = []
        if integer_indices is not None and len(integer_indices) > 0:
            assert integer_bounds is not None
            self.register_buffer(
                "integer_indices", torch.tensor(integer_indices, dtype=torch.long)
            )
            self.register_buffer("integer_bounds", integer_bounds)
            discrete_indices += integer_indices
        else:
            self.integer_indices = None
        self.categorical_features = categorical_features
        categorical_starts = []
        categorical_ends = []
        if self.categorical_features is not None:
            start = None
            for i, n_categories in categorical_features.items():
                if start is None:
                    start = i
                end = start + n_categories
                categorical_starts.append(start)
                categorical_ends.append(end)
                discrete_indices += list(range(start, end))
                start = end
        self.register_buffer(
            "discrete_indices", torch.tensor(discrete_indices, dtype=torch.long)
        )
        self.register_buffer(
            "categorical_starts", torch.tensor(categorical_starts, dtype=torch.long)
        )
        self.register_buffer(
            "categorical_ends", torch.tensor(categorical_ends, dtype=torch.long)
        )
        self.tau = tau
        # create cartesian product of discrete options
        discrete_options = []
        # add zeros for continuous params to simplify code
        for i in range(dim - len(discrete_indices)):
            discrete_options.append(
                torch.zeros(
                    1,
                    dtype=torch.long,
                )
            )
        if integer_bounds is not None:
            for i in range(integer_bounds.shape[-1]):
                discrete_options.append(
                    torch.arange(
                        integer_bounds[0, i], integer_bounds[1, i] + 1, dtype=torch.long
                    )
                )
        if categorical_features is not None:
            for cardinality in categorical_features.values():
                discrete_options.append(torch.arange(cardinality, dtype=torch.long))
        # categoricals are in numeric representation
        all_discrete_options = torch.cartesian_prod(*discrete_options)
        # one-hot encode the categoricals
        if categorical_features is not None and len(categorical_features) > 0:
            X_categ = torch.empty(
                *all_discrete_options.shape[:-1], sum(categorical_features.values())
            )
            i = 0
            for idx, cardinality in categorical_features.items():
                X_categ[..., i : i + cardinality] = one_hot(
                    all_discrete_options[..., idx],
                    num_classes=cardinality,
                ).to(X_categ)
                i = i + cardinality
            all_discrete_options = torch.cat(
                [all_discrete_options[..., : -len(categorical_features)], X_categ],
                dim=-1,
            )
        self.register_buffer("all_discrete_options", all_discrete_options)

    def get_rounding_prob(self, X: Tensor) -> Tensor:
        # todo consolidate this the MCProbabilisticReparameterizationInputTransform
        X_prob = X.detach().clone()
        if self.integer_indices is not None:
            # compute probabilities for integers
            X_int = X_prob[..., self.integer_indices]
            X_int_abs = X_int.abs()
            offset = X_int_abs.floor()
            if self.tau is not None:
                X_prob[..., self.integer_indices] = torch.sigmoid(
                    (X_int_abs - offset - 0.5) / self.tau
                )
            else:
                X_prob[..., self.integer_indices] = X_int_abs - offset
        # compute probabilities for categoricals
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            X_categ = X_prob[..., start:end]
            if self.tau is not None:
                X_prob[..., start:end] = torch.softmax(
                    (X_categ - 0.5) / self.tau, dim=-1
                )
            else:
                X_prob[..., start:end] = X_categ / X_categ.sum(dim=-1)
        return X_prob[..., self.discrete_indices]

    def get_probs(self, X: Tensor) -> Tensor:
        """
        Args:
            X: a `batch_shape x n x d`-dim tensor

        Returns:
            A `batch_shape x n_discrete x n`-dim tensors of probabilities of each discrete config under X.
        """
        # note this method should be differentiable
        X_prob = torch.ones(
            *X.shape[:-2],
            self.all_discrete_options.shape[0],
            X.shape[-2],
            dtype=X.dtype,
            device=X.device,
        )
        # n_discrete x batch_shape x n x d
        all_discrete_options = self.all_discrete_options.view(
            *([1] * (X.ndim - 2)), self.all_discrete_options.shape[0], *X.shape[-2:]
        ).expand(*X.shape[:-2], self.all_discrete_options.shape[0], *X.shape[-2:])
        X = X.unsqueeze(-3)
        if self.integer_indices is not None:
            # compute probabilities for integers
            X_int = X[..., self.integer_indices]
            X_int_abs = X_int.abs()
            offset = X_int_abs.floor()
            # note we don't actually need the sigmoid here
            X_prob_int = torch.sigmoid((X_int_abs - offset - 0.5) / self.tau)
            # X_prob_int = X_int_abs - offset
            for int_idx, idx in enumerate(self.integer_indices):
                offset_i = offset[..., int_idx]
                all_discrete_i = all_discrete_options[..., idx]
                diff = (offset_i + 1) - all_discrete_i
                round_up_mask = diff == 0
                round_down_mask = diff == 1
                neither_mask = ~(round_up_mask | round_down_mask)
                prob = X_prob_int[..., int_idx].expand(round_up_mask.shape)
                # need to be careful with in-place ops here for autograd
                X_prob[round_up_mask] = X_prob[round_up_mask] * prob[round_up_mask]
                X_prob[round_down_mask] = X_prob[round_down_mask] * (
                    1 - prob[round_down_mask]
                )
                X_prob[neither_mask] = X_prob[neither_mask] * 0

        # compute probabilities for categoricals
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            X_categ = X[..., start:end]
            X_prob_c = torch.softmax((X_categ - 0.5) / self.tau, dim=-1).expand(
                *X_categ.shape[:-3], all_discrete_options.shape[-3], *X_categ.shape[-2:]
            )
            for i in range(X_prob_c.shape[-1]):
                mask = all_discrete_options[..., start + i] == 1
                X_prob[mask] = X_prob[mask] * X_prob_c[..., i][mask]

        return X_prob

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        This is not sample-path differentiable.

        Args:
            X: A `batch_shape x 1 x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n_discrete x n x d`-dim tensor of rounded inputs.
        """
        n_discrete = self.discrete_indices.shape[0]
        all_discrete_options = self.all_discrete_options.view(
            *([1] * (X.ndim - 3)), self.all_discrete_options.shape[0], *X.shape[-2:]
        ).expand(*X.shape[:-3], self.all_discrete_options.shape[0], *X.shape[-2:])
        if X.shape[-1] > n_discrete:
            X = X.expand(
                *X.shape[:-3], self.all_discrete_options.shape[0], *X.shape[-2:]
            )
            return torch.cat(
                [X[..., :-n_discrete], all_discrete_options[..., -n_discrete:]], dim=-1
            )
        return self.all_discrete_options

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        # TODO: update this
        return super().equals(other=other) and torch.equal(
            self.integer_indices, other.integer_indices
        )


class MCProbabilisticReparameterizationInputTransform(InputTransform, Module):
    r"""Probabilistic reparameterization for ordinal and binary variables.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization).
    1. These are unnormalized back to the raw input space.
    2. The discrete ordinal valeus are sampled.
    3. All values are normalized to the unitcube.

    Example:
        >>> unnormalize_tf = Normalize(
        >>>     d=d,
        >>>     bounds=bounds,
        >>>     transform_on_eval=True,
        >>>     transform_on_train=True,
        >>>     reverse=True,
        >>> )
        >>> pr = OrdinalProbabilisticReparameterization(integer_indices)
        >>> normalize_tf = Normalize(d=d, bounds=bounds)
        >>> tf = ChainedInputTransform(
        >>>     tf1=unnormalize_tf, tf2=pr, tf3=normalize_tf
        >>> )
    """

    def __init__(
        self,
        integer_indices: Optional[List[int]] = None,
        integer_bounds: Optional[Tensor] = None,
        categorical_features: Optional[Dict[int, int]] = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        mc_samples: int = 128,
        resample: bool = False,
        flip: bool = False,
        tau: float = 0.1,
    ) -> None:
        r"""Initialize transform.

        Args:
            integer_indices: The indices of the integer inputs.
            categorical_features: The indices and cardinality of
                each categorical feature. The features are assumed
                to be one-hot encoded. TODO: generalize to support
                alternative representations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            mc_samples: The number of MC samples.
            resample: A boolean indicating whether to resample base samples
                at each forward pass.
            flip: A boolean indicating whether round based on u < p or 1 - p < u.
            tau: The temperature parameter.
        """
        super().__init__()
        if integer_indices is None and categorical_features is None:
            raise ValueError(
                "integer_indices and/or categorical_features must be provided."
            )
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        discrete_indices = []
        if integer_indices is not None and len(integer_indices) > 0:
            self.register_buffer(
                "integer_indices", torch.tensor(integer_indices, dtype=torch.long)
            )
            discrete_indices += integer_indices
        else:
            self.integer_indices = None
        self.categorical_features = categorical_features
        categorical_starts = []
        categorical_ends = []
        if self.categorical_features is not None:
            start = None
            for i, n_categories in categorical_features.items():
                if start is None:
                    start = i
                end = start + n_categories
                categorical_starts.append(start)
                categorical_ends.append(end)
                discrete_indices += list(range(start, end))
                start = end
        self.register_buffer(
            "discrete_indices", torch.tensor(discrete_indices, dtype=torch.long)
        )
        self.register_buffer(
            "categorical_starts", torch.tensor(categorical_starts, dtype=torch.long)
        )
        self.register_buffer(
            "categorical_ends", torch.tensor(categorical_ends, dtype=torch.long)
        )
        if integer_indices is None:
            self.register_buffer("integer_bounds", torch.tensor([], dtype=torch.long))
        else:
            self.register_buffer("integer_bounds", integer_bounds)
        self.mc_samples = mc_samples
        self.resample = resample
        self.flip = flip
        self.tau = tau

    def get_rounding_prob(self, X: Tensor) -> Tensor:
        X_prob = X.detach().clone()
        if self.integer_indices is not None:
            # compute probabilities for integers
            X_int = X_prob[..., self.integer_indices]
            X_int_abs = X_int.abs()
            offset = X_int_abs.floor()
            if self.tau is not None:
                X_prob[..., self.integer_indices] = torch.sigmoid(
                    (X_int_abs - offset - 0.5) / self.tau
                )
            else:
                X_prob[..., self.integer_indices] = X_int_abs - offset
        # compute probabilities for categoricals
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            X_categ = X_prob[..., start:end]
            if self.tau is not None:
                X_prob[..., start:end] = torch.softmax(
                    (X_categ - 0.5) / self.tau, dim=-1
                )
            else:
                X_prob[..., start:end] = X_categ / X_categ.sum(dim=-1)
        return X_prob[..., self.discrete_indices]

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        This is not sample-path differentiable.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of rounded inputs.
        """
        X_expanded = X.expand(*X.shape[:-3], self.mc_samples, *X.shape[-2:]).clone()
        X_prob = self.get_rounding_prob(X=X)
        if self.integer_indices is not None:
            X_int = X[..., self.integer_indices].detach()
            assert X.ndim > 1
            if X.ndim == 2:
                X.unsqueeze(-1)
            if (
                not hasattr(self, "base_samples")
                or self.base_samples.shape[-2:] != X_int.shape[-2:]
                or self.resample
            ):
                # construct sobol base samples
                bounds = torch.zeros(
                    2, X_int.shape[-1], dtype=X_int.dtype, device=X_int.device
                )
                bounds[1] = 1
                self.register_buffer(
                    "base_samples",
                    draw_sobol_samples(
                        bounds=bounds,
                        n=self.mc_samples,
                        q=X_int.shape[-2],
                        seed=torch.randint(0, 100000, (1,)).item(),
                    ),
                )
            X_int_abs = X_int.abs()
            # perform exact rounding
            is_negative = X_int < 0
            offset = X_int_abs.floor()
            prob = X_prob[..., : self.integer_indices.shape[0]]
            if self.flip:
                rounding_component = (1 - prob < self.base_samples).to(
                    dtype=X.dtype,
                )
            else:
                rounding_component = (prob >= self.base_samples).to(
                    dtype=X.dtype,
                )
            X_abs_rounded = offset + rounding_component
            X_int_new = (-1) ** is_negative.to(offset) * X_abs_rounded
            # clamp to bounds
            X_expanded[..., self.integer_indices] = torch.minimum(
                torch.maximum(X_int_new, self.integer_bounds[0]), self.integer_bounds[1]
            )

        # sample for categoricals
        if self.categorical_features is not None and len(self.categorical_features) > 0:
            if (
                not hasattr(self, "base_samples_categorical")
                or self.base_samples_categorical.shape[-2] != X.shape[-2]
                or self.resample
            ):
                bounds = torch.zeros(
                    2, len(self.categorical_features), dtype=X.dtype, device=X.device
                )
                bounds[1] = 1
                self.register_buffer(
                    "base_samples_categorical",
                    draw_sobol_samples(
                        bounds=bounds,
                        n=self.mc_samples,
                        q=X.shape[-2],
                        seed=torch.randint(0, 100000, (1,)).item(),
                    ),
                )

            # sample from multinomial as argmin_c [sample_c * exp(-x_c)]
            sample_d_start_idx = 0
            X_categ_prob = X_prob
            if self.integer_indices is not None:
                n_ints = self.integer_indices.shape[0]
                if n_ints > 0:
                    X_categ_prob = X_prob[..., n_ints:]

            for i, (idx, cardinality) in enumerate(self.categorical_features.items()):
                sample_d_end_idx = sample_d_start_idx + cardinality
                start = self.categorical_starts[i]
                end = self.categorical_ends[i]
                cum_prob = X_categ_prob[
                    ..., sample_d_start_idx:sample_d_end_idx
                ].cumsum(dim=-1)
                categories = (
                    (
                        (cum_prob > self.base_samples_categorical[..., i : i + 1])
                        .long()
                        .cumsum(dim=-1)
                        == 1
                    )
                    .long()
                    .argmax(dim=-1)
                )
                # one-hot encode
                X_expanded[..., start:end] = one_hot(
                    categories, num_classes=cardinality
                ).to(X)
                sample_d_start_idx = sample_d_end_idx

        return X_expanded

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            super().equals(other=other)
            and (self.resample == other.resample)
            and torch.equal(self.base_samples, other.base_samples)
            and (self.flip == other.flip)
            and torch.equal(self.integer_indices, other.integer_indices)
        )


@dataclass
class CategoricalSpec:
    idx: int
    num_categories: int


@dataclass
class LatentCategoricalSpec(CategoricalSpec):
    latent_dim: int


class EmbeddingTransform(InputTransform):
    r"""Abstract base class for Embedding-based transforms"""
    _emb_dim: int
    _transformed_dim: int
    dim: int
    categ_idcs: Tensor
    non_categ_mask: Tensor

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform categorical variables using embedding."""
        X_emb = torch.empty(
            X.shape[:-1] + torch.Size([self._emb_dim]), dtype=X.dtype, device=X.device
        )
        start_idx = 0
        for idx in self.categ_idcs.tolist():
            emb_table = self.get_emb_table(idx)
            emb_dim = emb_table.shape[-1]
            end_idx = start_idx + emb_dim
            emb = emb_table.index_select(dim=0, index=X[..., idx].reshape(-1).long())
            X_emb[..., start_idx:end_idx] = emb.view(
                X_emb.shape[:-1] + torch.Size([emb_dim])
            )
            start_idx = end_idx
        return torch.cat([X[..., self.non_categ_mask], X_emb], dim=-1)

    @abstractmethod
    def get_emb_table(self, idx: int) -> Tensor:
        r"""Get the embedding table for the specified categorical feature.

        Args:
            idx: The index of the categorical feature

        Returns:
            A `num_categories x emb_dim`-dim tensor containing the embeddings
            for each category.
        """
        pass

    def transform_bounds(self, bounds: Tensor) -> Tensor:
        r"""Update bounds based on embedding transform.

        Args:
            bounds: A `2 x d`-dim tensor of lower and upper bounds

        Returns:
            A x `2 x d_cont + d_emb`-dim tensor of lower and upper bounds
        """
        d_cont = self.dim - self.categ_idcs.shape[0]
        tf_bounds = torch.zeros(
            2, d_cont + self._emb_dim, dtype=bounds.dtype, device=bounds.device
        )
        tf_bounds[:, :d_cont] = bounds[:, self.non_categ_mask]
        tf_bounds[1, d_cont:] = 1
        return tf_bounds


class LatentCategoricalEmbedding(EmbeddingTransform, GPyTorchModule):
    r"""Latent embeddings for categorical variables.

    Note: this current uses the same latent embeddings across batched.
    This means that a batched multi-output model will use the same latent
    embeddings for all outputs.
    """

    def __init__(
        self,
        categorical_specs: List[LatentCategoricalSpec],
        dim: int,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_preprocess: bool = False,
        transform_on_fantasize: bool = False,
        eps: float = 1e-7,
    ) -> None:
        r"""Initialize input transform.

        Args:
            categorical_specs: A list of LatentCategoricalSpec objects.
            dim: the total dimension of the inputs.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: False
        """
        GPyTorchModule.__init__(self)
        self._eps = eps
        self.dim = dim
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess
        self.transform_on_fantasize = transform_on_fantasize
        self._emb_dim = 0
        categ_idcs = []
        # TODO: replace with ParameterDict when supported in GPyTorch
        for c in categorical_specs:
            nlzd_idx = normalize_indices([c.idx], dim)[0]
            categ_idcs.append(nlzd_idx)

            init_emb_table = draw_sobol_normal_samples(
                n=c.num_categories, d=c.latent_dim
            ).squeeze(0)
            self.register_parameter(
                f"raw_latent_emb_tables_{nlzd_idx}",
                nn.Parameter(init_emb_table),
            )

            def raw_latent_emb_table_setter(m, v, idx=nlzd_idx):
                m.initialize(f"raw_latent_emb_tables_{idx}", value=v)

            def raw_latent_emb_table_getter(m, idx=nlzd_idx):
                return getattr(m, f"raw_latent_emb_tables_{idx}")

            self._emb_dim += c.latent_dim
        self.register_buffer("categ_idcs", torch.tensor(categ_idcs, dtype=torch.long))
        non_categ_mask = torch.ones(dim, dtype=bool)
        non_categ_mask[self.categ_idcs] = 0
        self.register_buffer("non_categ_mask", non_categ_mask)

    def get_emb_table(self, idx: int) -> Tensor:
        r"""Get the embedding table for the specified categorical feature.

        Args:
            idx: The index of the categorical feature

        Returns:
            A `num_categories x latent_dim`-dim tensor containing the embeddings
            for each category.
        """
        # This technique is recommended in https://arxiv.org/abs/2003.03300
        raw_emb_table = getattr(self, f"raw_latent_emb_tables_{idx}")
        with torch.no_grad():
            raw_emb_table[0] = 0  # force one embedding to be the origin
            if raw_emb_table.shape[1] > 1 and raw_emb_table.shape[0] > 1:
                raw_emb_table[
                    1, 0
                ] = 0  # force one embedding to be on the x-axis if the embedding has two dimensions
        return raw_emb_table

    def untransform(
        self,
        X: Tensor,
        dist_func: Optional[Callable[[Tensor, Tensor, int], Tensor]] = None,
    ) -> Tensor:
        r"""Untransform X to represent categoricals as integers.

        The transformation assigns the category to be the index corresponding to the
        closest embedding. Note: this is not differentiable.

        Args:
            X: A `batch_shape x n x d_cont + d_latent`-dim tensor of transformed valiues
            dist_func: A broadcastable distance function mapping a two input tensors with
                shapes `batch_shape x n x 1 x d_latent` and `n_categories x d_latent` and
                an integer starting index to to a `batch_shape x n x n_categories`-dim
                tensor of distances. The default is L2 distance.

        Returns:
            The untransformed tensor.
        """
        new_X = torch.empty(
            X.shape[:-1] + torch.Size([self.dim]), dtype=X.dtype, device=X.device
        )
        num_non_categ_features = X.shape[-1] - self._emb_dim
        new_X[..., self.non_categ_mask] = X[..., :num_non_categ_features]
        start_idx = self.dim - self.categ_idcs.shape[0]
        for idx in self.categ_idcs.tolist():
            emb_table = self.get_emb_table(idx)
            emb_dim = emb_table.shape[-1]
            end_idx = start_idx + emb_dim
            x = X[..., start_idx:end_idx].unsqueeze(-2)
            x_emb = emb_table.unsqueeze(-3)
            if dist_func is not None:
                dist = dist_func(x, x_emb, start_idx)
            else:
                dist = torch.norm(x - x_emb, dim=-1)
            int_categories = dist.argmin(dim=-1).to(dtype=X.dtype)
            new_X[..., idx] = int_categories
            start_idx = end_idx
        return new_X
