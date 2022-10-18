#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

requirements = [
    "torch",
    "gpytorch",
    "botorch>=0.6",
    "scipy",
    "jupyter",
    "matplotlib",
    "nevergrad",
    "sklearn",
    "statsmodels",
    "xgboost",
]

dev_requires = [
    "black",
    "flake8",
    "pytest",
    "coverage",
]

setup(
    name="bo_pr",
    version="0.1",
    description="Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization",
    author="Anonymous Authors",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requires},
)
