#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Straight Through Estimators.
"""

import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
from torch.nn.functional import one_hot


class RoundSTE(Function):
    r"""Apply a rounding function and use a ST gradient estimator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
    ):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class OneHotArgmaxSTE(Function):
    r"""Apply a discretization (argmax) to a one-hot encoded categorical, return a one-hot encoded categorical, and use a STE gradient estimator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        num_categories: int,
    ):
        return one_hot(input.argmax(dim=-1), num_classes=num_categories)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class OneHotToNumericSTE(Function):
    r"""Apply an argmax function and use a STE gradient estimator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
    ):
        return input.argmax(dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
