#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for running experiments."""
from collections import OrderedDict
from botorch.models.transforms.input import Normalize, ChainedInputTransform
from botorch.acquisition.analytic import (
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
)
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.acquisition import AcquisitionFunction
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.sampling import (
    draw_sobol_samples,
)
from torch.nn.functional import one_hot
from itertools import product
from copy import deepcopy
from scipy.stats.mstats import winsorize as scipy_winsorize
from math import log
import torch
from botorch.models import FixedNoiseGP, SingleTaskGP, ModelListGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from discrete_mixed_bo.problems.base import (
    DiscretizedBotorchTestProblem,
    DiscreteTestProblem,
)
from discrete_mixed_bo.input import (
    LatentCategoricalEmbedding,
    LatentCategoricalSpec,
    Round,
    OneHotToNumeric,
)
from discrete_mixed_bo.model_utils import apply_normal_copula_transform
from discrete_mixed_bo.problems.binary import Contamination, LABS
from discrete_mixed_bo.problems.chemistry import Chemistry
from discrete_mixed_bo.problems.environmental import Environmental
from discrete_mixed_bo.problems.nashpobench2 import NASHPOBenchII
from discrete_mixed_bo.problems.re_problems import PressureVessel
from discrete_mixed_bo.problems.welded_beam import WeldedBeam
from discrete_mixed_bo.problems.svm import SVMFeatureSelection
from discrete_mixed_bo.problems.xgboost_hp import XGBoostHyperparameter
from discrete_mixed_bo.problems.pest import PestControl
from discrete_mixed_bo.problems.cco.cco import CCO
from discrete_mixed_bo.problems.oil_sorbent import OilSorbent, OilSorbentMixed
from discrete_mixed_bo.rffs import get_gp_sample_w_transforms
from discrete_mixed_bo.probabilistic_reparameterization import (
    AnalyticProbabilisticReparameterization,
    MCProbabilisticReparameterization,
)
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.likelihoods import GaussianLikelihood
from botorch.test_functions.synthetic import Ackley, Hartmann, Rosenbrock
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple, Union
from discrete_mixed_bo.kernels import get_kernel
from botorch.test_functions.multi_objective import DTLZ2

from discrete_mixed_bo.problems.coco_mixed_integer import Sphere
from statsmodels.distributions.empirical_distribution import ECDF


def eval_problem(X: Tensor, base_function: DiscreteTestProblem) -> Tensor:
    is_constrained = isinstance(base_function, ConstrainedBaseTestProblem)
    X_numeric = torch.zeros(
        *X.shape[:-1],
        base_function.bounds.shape[-1],
        dtype=X.dtype,
        device=X.device,
    )
    X_numeric[..., base_function.integer_indices] = X[
        ..., base_function.integer_indices
    ]
    X_numeric[..., base_function.cont_indices] = X[..., base_function.cont_indices]
    start_idx = None
    # X is one-hot encoded
    # transform from one-hot space to numeric space
    for i, cardinality in base_function.categorical_features.items():
        if start_idx is None:
            start_idx = i
        end_idx = start_idx + cardinality
        X_numeric[..., i] = (
            X[..., start_idx:end_idx].argmax(dim=-1).to(dtype=X_numeric.dtype)
        )
        start_idx = end_idx
    # normalize from integers to unit cube
    if len(base_function.categorical_features) > 0:
        X_numeric[..., base_function.categorical_indices] = normalize(
            X_numeric[..., base_function.categorical_indices],
            base_function.categorical_bounds,
        )
    X_numeric = unnormalize(X_numeric, base_function.bounds)
    Y = base_function(X_numeric)
    if Y.ndim == X_numeric.ndim - 1:
        Y = Y.unsqueeze(-1)

    if is_constrained:
        # here, non-negative Y_con implies feasibility
        Y_con = base_function.evaluate_slack(X_numeric)
        Y = torch.cat([Y, Y_con], dim=-1)
    return Y


def get_exact_rounding_func(
    bounds: Tensor,
    integer_indices: Optional[List[int]] = None,
    categorical_features: Optional[Dict[int, int]] = None,
    initialization: bool = False,
    return_numeric: bool = False,
    use_ste: bool = False,
) -> ChainedInputTransform:
    """Get an exact rounding function.

    The rounding function will take inputs from the unit cube,unnormalize them to the raw search space, round the inputs,
    and normalize them back to the unit cube.

    Categoricals are assumed to be one-hot encoded.

    Args:
        bounds: The raw search space bounds.
        integer_indices: The indices of the integer parameters
        categorical_features: A dictionary mapping indices to cardinalities for the categorical features.
        initialization: A boolean indication whether this exact rounding
        function is for initialization.
        return_numeric: a boolean indicating whether to return numeric or one-hot encoded categoricals
        use_ste: whether to use straight-through gradient estimation
    """
    if initialization:
        # this gives the extremes the same probability as the
        # interior values
        init_bounds = bounds.clone()
        init_bounds[0, integer_indices] -= 0.4999
        init_bounds[1, integer_indices] += 0.4999
    else:
        init_bounds = bounds

    tfs = OrderedDict()
    if integer_indices is not None and len(integer_indices) > 0:
        # unnormalize to integer space
        tfs["unnormalize_tf"] = Normalize(
            d=bounds.shape[1],
            bounds=init_bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=True,
        )
    # round
    tfs["round"] = Round(
        approximate=False,
        transform_on_train=False,
        transform_on_fantasize=False,
        # TODO: generalize
        integer_indices=integer_indices,
        categorical_features=categorical_features,
        use_ste=use_ste,
    )
    if integer_indices is not None and len(integer_indices) > 0:
        # renormalize to unit cube
        tfs["normalize_tf"] = Normalize(
            d=bounds.shape[1],
            bounds=bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=False,
        )
    if return_numeric:
        tfs["one_hot_to_numeric"] = OneHotToNumeric(
            categorical_features=categorical_features,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            use_ste=use_ste,
        )
    tf = ChainedInputTransform(**tfs)
    tf.to(dtype=bounds.dtype, device=bounds.device)
    tf.eval()
    return tf


def generate_initial_data(
    n: int,
    base_function: DiscreteTestProblem,
    bounds: Tensor,
    tkwargs: dict,
    init_exact_rounding_func: ChainedInputTransform,
) -> Tuple[Tensor, Tensor]:
    r"""
    Generates the initial data for the experiments.
    Args:
        n: Number of training points.
        base_function: The base problem.
        bounds: The bounds to generate the training points from. `2 x d`-dim tensor.
        tkwargs: Arguments for tensors, dtype and device.
        init_exact_rounding_func: The exact rounding function for initialization.

    Returns:
        The train_X and train_Y. `n x d` and `n x m`.
    """
    raw_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2).to(**tkwargs)
    train_x = init_exact_rounding_func(raw_x)
    train_obj = eval_problem(train_x, base_function=base_function)
    return train_x, train_obj


def apply_winsorize(
    y: Tensor, winsorization_level: float, maximize: bool = True
) -> Tensor:
    if maximize:
        winsorize_limits = (winsorization_level, None)
    else:
        winsorize_limits = (None, winsorization_level)
    return torch.from_numpy(scipy_winsorize(y.cpu().numpy(), winsorize_limits)).to(y)


def initialize_model(
    train_x: Tensor,
    train_y: Tensor,
    binary_dims: List[int],
    categorical_features: Optional[List[int]] = None,
    use_model_list: bool = False,
    use_fixed_noise: bool = True,
    kernel_type: str = "mixed",
    copula: bool = False,
    winsorize: bool = False,
    latent_emb_dim: Optional[int] = None,
    use_ard_binary: bool = False,
    function_name: Optional[str] = None,
) -> Tuple[
    Union[ExactMarginalLogLikelihood, SumMarginalLogLikelihood],
    Union[FixedNoiseGP, SingleTaskGP, ModelListGP],
    Optional[List[ECDF]],
]:
    r"""Constructs the model and its MLL.

    TODO: add better kernel selection for binary inputs.

    Args:
        train_x: An `n x d`-dim tensor of training inputs.
        train_y: An `n x m`-dim tensor of training outcomes.
        use_model_list: If True, returns a ModelListGP with models for each outcome.
        use_fixed_noise: If True, assumes noise-free outcomes and uses FixedNoiseGP.
    Returns:
        The MLL and the model. Note: the model is not trained!
    """
    base_model_class = FixedNoiseGP if use_fixed_noise else SingleTaskGP
    # define models for objective and constraint
    if copula:
        train_y, ecdfs = apply_normal_copula_transform(train_y)
    else:
        ecdfs = None
    if winsorize:
        train_y = apply_winsorize(train_y, winsorization_level=0.2)
    if use_fixed_noise:
        train_Yvar = torch.full_like(train_y, 1e-7) * train_y.std(dim=0).pow(2)
    if kernel_type in ("mixed_categorical", "mixed_latent"):
        # map one-hot categoricals to numeric representation
        input_transform = OneHotToNumeric(categorical_features=categorical_features)
        input_transform.eval()
        train_x = input_transform(train_x)
    if categorical_features is None or kernel_type == "mixed_latent":
        categorical_dims = []
    else:
        categorical_dims = list(categorical_features.keys())
    categorical_transformed_features = categorical_features
    model_kwargs = []
    for i in range(train_y.shape[-1]):
        transformed_x = train_x
        if kernel_type == "mixed_latent":
            categorical_transformed_features = {}
            specs = []
            start = None
            for idx, card in categorical_features.items():
                if start is None:
                    start = idx
                spec = LatentCategoricalSpec(
                    idx=idx,
                    num_categories=card,
                    latent_dim=latent_emb_dim
                    if latent_emb_dim is not None
                    else (1 if card <= 3 else 2),
                )
                categorical_transformed_features[start] = spec.latent_dim
                start = start + spec.latent_dim
                specs.append(spec)

            input_transform = LatentCategoricalEmbedding(
                specs,
                dim=train_x.shape[-1],
            ).to(train_x)
            with torch.no_grad():
                transformed_x = input_transform(train_x)
            cat_start = train_x.shape[-1] - len(categorical_features)
            categorical_dims = list(range(cat_start, transformed_x.shape[-1]))
        else:
            input_transform = None

        model_kwargs.append(
            {
                "train_X": train_x,
                "train_Y": train_y[..., i : i + 1],
                "outcome_transform": None if copula else Standardize(m=1),
                "covar_module": get_kernel(
                    kernel_type=kernel_type,
                    dim=transformed_x.shape[-1],
                    binary_dims=binary_dims,
                    categorical_transformed_features=categorical_transformed_features,
                    train_X=train_x,
                    train_Y=train_y,
                    function_name=function_name,
                    use_ard_binary=use_ard_binary,
                ),
                "input_transform": input_transform,
            }
        )
        if use_fixed_noise:
            model_kwargs[i]["train_Yvar"] = train_Yvar[..., i : i + 1]
        else:
            model_kwargs[i]["likelihood"] = GaussianLikelihood(
                noise_prior=GammaPrior(0.9, 10.0),
                noise_constraint=Interval(1e-7, 1e-3),
            )

    models = [base_model_class(**model_kwargs[i]) for i in range(train_y.shape[-1])]
    if len(models) > 1:
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        model = models[0]
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model, ecdfs


def get_EI(
    model: Model,
    train_Y: Tensor,
    num_constraints: int,
    posterior_transform: Optional[ScalarizedPosteriorTransform] = None,
) -> ExpectedImprovement:
    if posterior_transform is not None:
        obj = posterior_transform.evaluate(train_Y)
    else:
        obj = train_Y[..., 0]
    if num_constraints > 0:
        feas = (train_Y[..., 1:] >= 0).all(dim=-1)
        if feas.any():
            best_f = obj[feas].max()
        else:
            # take worst point
            best_f = -obj.max()
        if posterior_transform is not None:
            raise NotImplementedError
        return ConstrainedExpectedImprovement(
            model=model,
            best_f=best_f,
            objective_index=0,
            constraints={i: (0.0, None) for i in range(1, num_constraints + 1)},
        )
    return ExpectedImprovement(
        model, best_f=obj.max(), posterior_transform=posterior_transform
    )


def get_EHVI(
    model: Model, train_Y: Tensor, ref_point: Tensor
) -> ExpectedHypervolumeImprovement:
    bd = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)
    return ExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=bd,
    )


def get_qEHVI(
    model: Model, train_Y: Tensor, ref_point: Tensor
) -> qExpectedHypervolumeImprovement:
    bd = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)
    return qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=bd,
    )


def get_acqf(
    label: str,
    mc_samples: int,
    model: GPyTorchModel,
    X_baseline: Tensor,
    train_Y: Tensor,
    iteration: int,
    tkwargs: dict,
    num_constraints: int,
    base_function: DiscreteTestProblem,
    exact_rounding_func: ChainedInputTransform,
    batch_size: int = 1,
    **kwargs,
) -> Union[AcquisitionFunction, List[AcquisitionFunction]]:
    r"""Combines a few of the above utils to construct the acqf."""
    is_constrained = isinstance(base_function, ConstrainedBaseTestProblem)
    if is_constrained and label[-2:] != "ei":
        raise NotImplementedError("Only EI is currently supported with constraints.")
    num_constraints = base_function.num_constraints if is_constrained else 0
    if base_function.objective_weights is not None:
        posterior_transform = ScalarizedPosteriorTransform(
            weights=base_function.objective_weights
        )
    else:
        posterior_transform = None
    if label[-3:] == "ucb":
        beta = 0.2 * X_baseline.shape[-1] * log(2 * iteration)
    if ("exact_round" in label) or ("enumerate" in label):
        if isinstance(model, ModelListGP):
            models = model.models
            for m in model.models:
                if hasattr(m, "input_transform"):
                    m.input_transform = ChainedInputTransform(
                        round=deepcopy(exact_rounding_func), orig=m.input_transform
                    )
                else:
                    m.input_transform = deepcopy(exact_rounding_func)
        else:
            if hasattr(model, "input_transform"):
                model.input_transform = ChainedInputTransform(
                    round=exact_rounding_func, orig=model.input_transform
                )
            else:
                model.input_transform = exact_rounding_func
    if batch_size == 1:
        if label[-2:] == "ei":
            acq_func = get_EI(
                model=model,
                train_Y=train_Y,
                num_constraints=num_constraints,
                posterior_transform=posterior_transform,
            )

        elif label[-3:] == "ucb":
            acq_func = UpperConfidenceBound(
                model=model,
                beta=beta,
                posterior_transform=posterior_transform,
            )
        elif label[-2:] == "ts" or label[-7:] == "nehvi-1":
            model = get_gp_sample_w_transforms(
                model=model,
                num_outputs=model.num_outputs,
                n_samples=1,
                num_rff_features=kwargs.get("num_rff_features", 512),
            )
            if label[-2:] == "ts":
                acq_func = PosteriorMean(
                    model=model,
                    posterior_transform=posterior_transform,
                )
            if label[-7:] == "nehvi-1":
                with torch.no_grad():
                    preds = model.posterior(X_baseline).mean
                acq_func = get_qEHVI(
                    model=model,
                    train_Y=preds,
                    ref_point=base_function.ref_point,
                )
        elif label[-4:] == "ehvi":
            acq_func = get_EHVI(
                model=model,
                train_Y=train_Y,
                ref_point=base_function.ref_point,
            )
        else:
            raise NotImplementedError
    if "pr" in label:
        if kwargs.get("pr_use_analytic", False):
            acq_func = AnalyticProbabilisticReparameterization(
                acq_function=acq_func,
                dtype=train_Y.dtype,
                device=train_Y.device,
                integer_indices=base_function.integer_indices.cpu().tolist(),
                integer_bounds=base_function.integer_bounds,
                categorical_features=base_function.categorical_features,
                dim=X_baseline.shape[-1],
                apply_numeric=kwargs.get("apply_numeric", False),
                tau=kwargs.get("pr_tau", 0.1),
            )
        else:
            acq_func = MCProbabilisticReparameterization(
                acq_function=acq_func,
                integer_indices=base_function.integer_indices.cpu().tolist(),
                integer_bounds=base_function.integer_bounds,
                categorical_features=base_function.categorical_features,
                dim=X_baseline.shape[-1],
                batch_limit=kwargs.get("pr_batch_limit", 32),
                mc_samples=kwargs.get("pr_mc_samples", 1024),
                apply_numeric=kwargs.get("apply_numeric", False),
                tau=kwargs.get("pr_tau", 0.1),
            )
    return acq_func


def get_problem(name: str, dim: Optional[int] = None, **kwargs) -> DiscreteTestProblem:
    r"""Initialize the test function."""
    if name == "discrete_hartmann":
        dim = 6
        integer_bounds = torch.zeros(2, 4)
        integer_bounds[1, :2] = 2  # 3 values
        integer_bounds[1, 2:4] = 9  # 10 values
        hartmann = Hartmann(dim=dim, negate=True)
        return DiscretizedBotorchTestProblem(
            problem=hartmann,
            integer_indices=list(range(4)),
            integer_bounds=integer_bounds,
        )
    elif name == "discrete_hartmann2":
        dim = 6
        integer_bounds = torch.zeros(2, 4)
        integer_bounds[1, :2] = 3  # 3 values
        integer_bounds[1, 2:4] = 19  # 10 values
        hartmann = Hartmann(dim=dim, negate=True)
        return DiscretizedBotorchTestProblem(
            problem=hartmann,
            integer_indices=list(range(2, 6)),
            integer_bounds=integer_bounds,
        )
    elif name == "categorical_hartmann":
        dim = 6
        categorical_bounds = torch.zeros(2, 2)
        categorical_bounds[1, :2] = 2  # 3 values
        hartmann = Hartmann(dim=dim, negate=True)
        return DiscretizedBotorchTestProblem(
            problem=hartmann,
            categorical_indices=list(range(4, 6)),
            categorical_bounds=categorical_bounds,
        )
    elif name == "discrete_ackley":
        dim = 20
        integer_bounds = torch.zeros(2, dim - 5)
        integer_bounds[1, :5] = 2  # 3 values
        integer_bounds[1, 5:10] = 4  # 5 values
        integer_bounds[1, 10:15] = 9  # 10 values
        ackley = Ackley(dim=dim, negate=True)
        return DiscretizedBotorchTestProblem(
            problem=ackley,
            integer_indices=list(range(dim - 5)),
            integer_bounds=integer_bounds,
        )
    elif name == "ackley13":
        dim = 13
        ackley = Ackley(dim=dim, negate=True)
        ackley.bounds[0, :-3] = 0
        ackley.bounds[1] = 1
        ackley.bounds[0, -3:] = -1
        integer_bounds = torch.zeros(2, dim - 3)
        integer_bounds[1] = 1  # 2 values
        return DiscretizedBotorchTestProblem(
            problem=ackley,
            integer_indices=list(range(dim - 3)),
            integer_bounds=integer_bounds,
        )
    elif name == "integer_ackley13":
        dim = 13
        ackley = Ackley(dim=dim, negate=True)
        ackley.bounds[0, :-3] = 0
        ackley.bounds[1] = 1
        ackley.bounds[0, -3:] = -1
        integer_bounds = torch.zeros(2, dim - 3)
        integer_bounds[1, :5] = 2  # 3 values
        integer_bounds[1, dim - 8 :] = 4  # 5 values
        return DiscretizedBotorchTestProblem(
            problem=ackley,
            integer_indices=list(range(dim - 3)),
            integer_bounds=integer_bounds,
        )
    elif name == "contamination":
        assert dim is not None
        return Contamination(dim=dim, negate=True)
    elif name == "labs":
        assert dim is not None
        return LABS(dim=dim, negate=True)
    elif name == "svm":
        data = kwargs.get("data")
        assert data is not None
        assert dim is not None
        return SVMFeatureSelection(
            data=data,
            dim=dim,
            negate=True,
        )
    elif name == "discrete_oil":
        return OilSorbent(negate=True)
    elif name == "mixed_oil":
        return OilSorbentMixed(negate=True)
    elif name == "cco":
        return CCO(
            data=kwargs.get("data"),
            negate=True,
            scalarize=kwargs.get("scalarize", False),
        )

    elif name == "discrete_dtlz2":
        dim = 6
        integer_bounds = torch.zeros(2, 4)
        integer_bounds[1, :1] = 10  # 100 values
        integer_bounds[1, 1:4] = 4  # 5 values
        dtlz2 = DTLZ2(dim=dim, negate=True)
        return DiscretizedBotorchTestProblem(
            problem=dtlz2,
            integer_indices=list(range(4)),
            integer_bounds=integer_bounds,
        )
    elif name == "environmental":
        return Environmental(negate=True)
    elif name == "mixed_int_f1":
        integer_bounds = torch.zeros(2, 8)
        integer_bounds[1, :2] = 1
        integer_bounds[1, 2:4] = 2
        integer_bounds[1, 4:6] = 4
        integer_bounds[1, 6:] = 6
        return DiscretizedBotorchTestProblem(
            problem=Sphere(negate=True, dim=16),
            integer_indices=list(range(8)),
            integer_bounds=integer_bounds,
        )
    elif name == "mixed_int_f3":
        integer_bounds = torch.zeros(2, 16)
        integer_bounds[1, :4] = 1
        integer_bounds[1, 4:8] = 3
        integer_bounds[1, 8:12] = 7
        integer_bounds[1, 12:16] = 15
        return DiscretizedBotorchTestProblem(
            problem=Sphere(negate=True, dim=20),
            integer_indices=list(range(16)),
            integer_bounds=integer_bounds,
        )
    elif name == "nashpobench2":
        data = kwargs.get("data")
        assert data is not None
        return NASHPOBenchII(
            data=data,
            negate=True,
            num_objectives=kwargs.get("num_objectives", 1),
            use_12_epoch_result=kwargs.get("use_12_epoch_result", False),
            use_log=kwargs.get("use_log", False),
        )
    elif name == "pressure_vessel":
        return PressureVessel(negate=True)
    elif name == "rosenbrock10":
        rosen = Rosenbrock(dim=10, negate=True)
        integer_bounds = torch.zeros(2, 6)
        integer_bounds[1, :] = 2  # 3 values
        return DiscretizedBotorchTestProblem(
            problem=rosen,
            integer_indices=list(range(4, 10)),
            integer_bounds=integer_bounds,
        )
    elif name == "rosenbrock10_scaled":
        rosen = Rosenbrock(dim=10, negate=True)
        rosen.bounds[0, :6] = -2.1
        rosen.bounds[1, :6] = 2.3
        rosen.bounds[0, 6:] = -2
        rosen.bounds[1, 6:] = 2
        integer_bounds = torch.zeros(2, 6)
        integer_bounds[1, :] = 3  # 4 values
        return DiscretizedBotorchTestProblem(
            problem=rosen,
            integer_indices=list(range(6)),
            integer_bounds=integer_bounds,
        )
    elif name == "pest":
        return PestControl(
            dim=kwargs.get("dim", 25), n_choice=kwargs.get("n_choice", 5), negate=True
        )
    elif name == "xgb":
        return XGBoostHyperparameter(
            task=kwargs.get("task", "mnist"),
            negate=True,
            data=kwargs.get("data"),
        )
    elif name == "chemistry":
        return Chemistry(
            negate=True,
            data=kwargs.get("data"),
        )
    elif name == "welded_beam":
        return WeldedBeam(
            negate=True,
            continuous=kwargs.get("continuous", False),
        )
    else:
        raise ValueError(f"Unknown function name: {name}!")


def generate_discrete_options(
    base_function: DiscreteTestProblem,
) -> List[Dict[int, float]]:
    categorical_features = base_function.categorical_features
    discrete_indices = torch.cat(
        [base_function.integer_indices, base_function.categorical_indices], dim=0
    )
    cardinalities = (
        (
            base_function.bounds[1, discrete_indices]
            - base_function.bounds[0, discrete_indices]
            + 1
        )
        .long()
        .tolist()
    )
    discrete_indices_list = discrete_indices.tolist()
    discrete_options = torch.tensor(
        list(product(*[range(c) for c in cardinalities])),
        dtype=torch.long,
    )
    indices = base_function.integer_indices.tolist()
    # now one-hot encode the categoricals
    if categorical_features is not None:
        one_hot_categoricals = [
            one_hot(discrete_options[:, i], num_classes=cardinalities[i])
            for i in range(
                base_function.integer_indices.shape[0], discrete_options.shape[1]
            )
        ]
        discrete_options = torch.cat(
            [
                discrete_options[:, : base_function.integer_indices.shape[0]],
                *one_hot_categoricals,
            ],
            dim=-1,
        )

        # get a list of the starting and ending indices of each categorical feature in one-hot space
        start_idx = None
        for i in sorted(categorical_features.keys()):
            if start_idx is None:
                start_idx = i
            end_idx = start_idx + categorical_features[i]
            categ_indices = list(range(start_idx, end_idx))
            indices.extend(categ_indices)
            start_idx = end_idx
    # create a list of dictionaries of mapping indices to values
    # the list has a dictionary for each discrete configuration
    return [dict(zip(xi, indices)) for xi in discrete_options.tolist()]
