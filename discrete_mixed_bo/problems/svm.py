#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.svm import SVR
from torch import Tensor
from xgboost import XGBRegressor

from discrete_mixed_bo.problems.base import DiscreteTestProblem


def process_uci_data(
    data: np.ndarray, n_features: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The slice dataset can be downloaded from: https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis

    # Get the input data
    X = data[:, :-1]
    X -= X.min(axis=0)
    X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
    X = X / (X.max(axis=0) - X.min(axis=0))
    X = 2 * X - 1
    assert X.min() == -1 and X.max() == 1

    # Standardize targets
    y = data[:, -1]
    y = (y - y.mean()) / y.std()

    # Only keep 10,000 data points and n_features features
    shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[
        :10000
    ]  # Use seed 0
    X, y = X[shuffled_indices], y[shuffled_indices]

    # Use Xgboost to figure out feature importances and keep only the most important features
    xgb = XGBRegressor(max_depth=8, random_state=0).fit(X, y)
    inds = (-xgb.feature_importances_).argsort()
    X = X[:, inds[:n_features]]

    # Train/Test split on a subset of the data
    train_n = int(math.floor(0.50 * X.shape[0]))
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:], y[train_n:]

    return train_x, train_y, test_x, test_y


class SVMFeatureSelection(DiscreteTestProblem):
    def __init__(
        self,
        dim: int,
        data: np.ndarray,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        n_features = dim - 3
        self.train_x, self.train_y, self.test_x, self.test_y = process_uci_data(
            data=data, n_features=n_features
        )
        self.n_features = n_features
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(
            negate=negate, noise_std=noise_std, integer_indices=list(range(n_features))
        )

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.tensor(
            [self._evaluate_true(x.numpy()) for x in X.view(-1, self.dim).cpu()],
            dtype=X.dtype,
            device=X.device,
        ).view(X.shape[:-1])

    def _evaluate_true(self, x: np.ndarray):
        assert x.shape == (self.dim,)
        assert (x >= self.bounds[0].cpu().numpy()).all() and (
            x <= self.bounds[1].cpu().numpy()
        ).all()
        assert (
            (x[: self.n_features] == 0) | (x[: self.n_features] == 1)
        ).all()  # Features must be 0 or 1
        inds_selected = np.where(x[: self.n_features] == 1)[0]
        if inds_selected.shape[0] == 0:
            # if no features, use the mean prediction
            pred = self.train_y.mean(axis=0)
        else:
            epsilon = 0.01 * 10 ** (2 * x[-3])  # Default = 0.1
            C = 0.01 * 10 ** (4 * x[-2])  # Default = 1.0
            gamma = (
                (1 / self.n_features) * 0.1 * 10 ** (2 * x[-1])
            )  # Default = 1.0 / self.n_features
            model = SVR(C=C, epsilon=epsilon, gamma=gamma)
            model.fit(self.train_x[:, inds_selected], self.train_y)
            pred = model.predict(self.test_x[:, inds_selected])
        mse = ((pred - self.test_y) ** 2).mean(axis=0)
        return 1 * math.sqrt(mse)  # Return RMSE
