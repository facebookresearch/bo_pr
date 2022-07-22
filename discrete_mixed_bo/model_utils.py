#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from statsmodels.distributions.empirical_distribution import ECDF
from torch import Tensor
from torch.distributions import Normal


def apply_normal_copula_transform(
    Y: Tensor, ecdfs: Optional[List[ECDF]] = None
) -> Tuple[Tensor, List[ECDF]]:
    r"""Apply a copula transform independently to each output.

    Values are first mapped to quantiles through the empirical cdf, then
    through an inverse standard normal cdf.

    Note: this is not currently differentiable and it does not support
    batched `Y`.

    TODO: Remove dependency on ECDF or at least write an abstract specification
    of what we expect ECDF to do.

    Args:
        Y: A `n x m`-dim tensor of values
        ecdfs: A list of ecdfs to use in the transformation

    Returns:
        2-element tuple containing

        - A `n x m`-dim tensor of transformed values
        - A list of `m` ECDF objects.
    """
    if Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")
    normal = Normal(0, 1)
    Y_i_tfs = []
    ecdfs = ecdfs or []
    for i in range(Y.shape[-1]):
        Y_i = Y[:, i].cpu().numpy()
        if len(ecdfs) <= i:
            # compute new ecdf if None were provided
            ecdf = ECDF(Y_i)
            ecdfs.append(ecdf)
        else:
            # Otherwise use existing ecdf
            ecdf = ecdfs[i]
        # clamp quantiles here to avoid (-)infs at the extremes
        Y_i_tf = normal.icdf(torch.from_numpy(ecdf(Y_i)).to(Y).clamp(0.0001, 0.9999))
        Y_i_tfs.append(Y_i_tf)
    return torch.stack(Y_i_tfs, dim=-1), ecdfs
