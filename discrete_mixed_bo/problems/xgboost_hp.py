#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from math import exp, log
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import xgboost
from sklearn import datasets, metrics, model_selection

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class XGBoostHyperparameter(DiscreteTestProblem):
    dim: int = 11

    def __init__(
        self,
        task="mnist",
        split=0.3,
        seed=None,
        negate: bool = False,
        noise_std: Optional[float] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        The XGboost hyperparameter tuning task on MNIST classification
        Args:
            task: 'mnist' or 'boston'
            split: train-test split
            normalize:
            seed:
            negate:
            noise_std:

        We optimize the following hyperparameters, in the order presented below:
        Categoricals:
            0. booster type -- cat (2 choices) -- gbtree, dart
            1. grow policy -- cat (2 choices) -- depthwise, lossguide
            2. training objective -- cat (2 choices) -- softmax, softprob, 3 choices for regression ['reg:linear',  'reg:gamma', 'reg:tweedie']
        Integers:
            3. max depth -- int -- [1, 10]
            4. min_child_weight: -- uniform int -- [1,10]
        Floats:
            5. log10-learning rate -- uniform float -- [-5, 0]
            6. gamma -- uniform float -- [0, 10]
            7. subsample -- log float -- [0.1, 1]
            8. lambda (L2 regularization weight) -- log float -- [1e-3, 5]
            9. alpha (L1 regularization weight) -- log float -- [1e-3, 5]
            10. colsample_by_tree -- uniform float -- (0, 1]
        """
        self.task = task
        self.split = split
        self.seed = seed
        if task == "airfoil":
            if data is not None:
                # data comes from https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat
                self.reg_or_clf = "reg"
                self.data = data
        if data is None:
            self.data, self.reg_or_clf = get_data_and_task_type(self.task)
        stratify = self.data["target"] if self.reg_or_clf == "clf" else None

        if self.reg_or_clf == "clf":
            self._bounds = [
                (0, 1),
                (0, 1),
                (0, 1),
                (3, 15),
                (100, 500),
                (-5, -1),
                (0, 10),
                (log(0.1), log(1)),
                (log(1e-3), log(5)),
                (log(1e-3), log(5)),
                (0, 1),
            ]
        else:
            self._bounds = [
                (0, 1),
                (0, 1),
                (0, 2),
                (1, 10),
                (1, 10),
                (-5, -1),
                (0, 10),
                (0.1, 1),
                (0, 5),
                (0, 5),
                (0.3, 1),
            ]
        super(XGBoostHyperparameter, self).__init__(
            negate=negate,
            noise_std=noise_std,
            categorical_indices=[2],
            integer_indices=[0, 1, 3, 4],
        )
        (
            self.train_x,
            self.test_x,
            self.train_y,
            self.test_y,
        ) = model_selection.train_test_split(
            self.data["data"],
            self.data["target"],
            test_size=self.split,
            stratify=stratify,
            random_state=self.seed,
        )

    def _evaluate_single(self, x: torch.Tensor):
        model = self.create_model(x)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)
        # 1-acc for minimization
        if self.reg_or_clf == "clf":
            score = 1 - metrics.accuracy_score(self.test_y, y_pred)
        elif self.reg_or_clf == "reg":
            score = metrics.mean_squared_error(self.test_y, y_pred)
        else:
            raise NotImplementedError

        return torch.tensor(score, dtype=torch.float)

    def _parse_input(self, x) -> dict:
        """Parse the input into a dictionary"""
        kwargs = {}
        x = x.detach().numpy()
        assert len(x) == self.dim
        args = [
            "booster",
            "grow_policy",
            "objective",
            "max_depth",
            "min_child_weight",
            "learning_rate",
            "gamma",
            "subsample",
            "reg_lambda",
            "reg_alpha",
            "colsample_bytree",
        ]
        for i, val in enumerate(x):
            if args[i] == "booster":
                kwargs[args[i]] = ["gbtree", "dart"][int(val)]
            elif args[i] == "grow_policy":
                kwargs[args[i]] = ["depthwise", "lossguide"][int(val)]
            elif args[i] == "objective":
                if self.reg_or_clf == "clf":
                    kwargs[args[i]] = ["multi:softmax", "multi:softprob"][int(val)]
                else:
                    kwargs[args[i]] = ["reg:linear", "reg:gamma", "reg:tweedie"][
                        int(val)
                    ]
            elif args[i] == "learning_rate":
                kwargs[args[i]] = float(10**val)
            elif args[i] in ("subsample", "reg_lambda", "reg_alpha"):
                kwargs[args[i]] = float(val)
            elif args[i] in ["max_depth", "min_child_weight", "n_estimators"]:
                kwargs[args[i]] = int(val)
            else:
                kwargs[args[i]] = float(val)
        # print(kwargs)
        return kwargs

    def create_model(self, x):
        xgboost_kwargs = self._parse_input(x)
        if self.reg_or_clf == "clf":
            model = xgboost.XGBClassifier(
                eval_metric="mlogloss", **xgboost_kwargs, seed=self.seed, n_jobs=10
            )
        else:
            model = xgboost.XGBRegressor(**xgboost_kwargs, seed=self.seed, n_jobs=10)
        return model

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        res = (
            torch.stack(
                [self._evaluate_single(x) for x in X.cpu().view(-1, self.dim)],
            )
            .to(X)
            .view(*X.shape[:-1])
        )
        return res


def get_data_and_task_type(task):
    if task == "boston":
        data = datasets.load_boston()
        reg_or_clf = "reg"  # regression or classification
    elif task == "diabetes":
        data = datasets.load_diabetes()
        reg_or_clf = "reg"  # regression or classification
    elif task == "airfoil":
        # data comes from https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "data", "airfoil_self_noise.dat")
        df = pd.read_csv(path, header=None, sep="\t")
        data = {"data": df.iloc[:, :5].values, "target": df.iloc[:, 5].values}
        reg_or_clf = "reg"
    elif task == "mnist":
        data = datasets.load_digits()
        reg_or_clf = "clf"
    else:
        raise NotImplementedError("Bad choice for task")
    return data, reg_or_clf
