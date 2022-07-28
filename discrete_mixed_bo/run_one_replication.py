#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Run one replication.
"""
import nevergrad as ng
from time import time
import torch
from torch import Tensor
from typing import Callable, Optional, List, Dict
from discrete_mixed_bo.experiment_utils import (
    eval_problem,
    generate_initial_data,
    initialize_model,
    generate_discrete_options,
    get_acqf,
    get_exact_rounding_func,
    get_problem,
)
from discrete_mixed_bo.input import OneHotToNumeric
from discrete_mixed_bo.model_utils import apply_normal_copula_transform
from discrete_mixed_bo.optimize import optimize_acqf
from botorch.optim.optimize import optimize_acqf_discrete, optimize_acqf_mixed
import numpy as np
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from discrete_mixed_bo.trust_region import TurboState, update_state
from botorch.fit import fit_gpytorch_model
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.acquisition.utils import is_nonnegative
import gc

supported_labels = [
    "sobol",
    "cont_optim__round_after__ei",
    "pr__ei",
    "exact_round__fin_diff__ei",
    "exact_round__ste__ei",
    "enumerate__ei",
    "cont_optim__round_after__ts",
    "pr__ts",
    "exact_round__fin_diff__ts",
    "exact_round__ste__ts",
    "enumerate__ts",
    "cont_optim__round_after__ucb",
    "pr__ucb",
    "exact_round__fin_diff__ucb",
    "exact_round__ste__ucb",
    "enumerate__ucb",
    "cont_optim__round_after__ehvi",
    "pr__ehvi",
    "exact_round__fin_diff__ehvi",
    "exact_round__ste__ehvi",
    "enumerate__ehvi",
    "cont_optim__round_after__nehvi-1",
    "pr__nehvi-1",
    "exact_round__fin_diff__nehvi-1",
    "exact_round__ste__nehvi-1",
    "enumerate__nehvi-1",
    "nevergrad_portfolio",
]


def run_one_replication(
    seed: int,
    label: str,
    iterations: int,
    function_name: str,
    batch_size: int,
    mc_samples: int,
    n_initial_points: Optional[int] = None,
    optimization_kwargs: Optional[dict] = None,
    dim: Optional[int] = None,
    acqf_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
    save_frequency: Optional[int] = None,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
    save_callback: Optional[Callable[[Tensor], None]] = None,
    problem_kwargs: Optional[Dict[str, np.ndarray]] = None,
    use_trust_region: bool = False,
) -> None:
    r"""Run the BO loop for given number of iterations. Supports restarting of
    prematurely killed experiments.
    Args:
        seed: The experiment seed.
        label: The label / algorithm to use.
        iterations: Number of iterations of the BO loop to perform.
        n_initial_points: Number of initial evaluations to use.
        function_name: The name of the test function to use.
        batch_size: The q-batch size, i.e., number of parallel function evaluations.
        mc_samples: Number of MC samples used for MC acquisition functions (e.g., NEI).
        optimization_kwargs: Arguments passed to `optimize_acqf`. Includes `num_restarts`
            and `raw_samples` and other optional arguments.
        model_kwargs: Arguments for `initialize_model`. The default behavior is to use
            a ModelListGP consisting of noise-free FixedNoiseGP models.
        save_frequency: How often to save the output.
        dtype: The tensor dtype to use.
        device: The device to use.
        save_callback: method to save results to file
    """
    assert label in supported_labels, "Label not supported!"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkwargs = {"dtype": dtype, "device": device}
    model_kwargs = model_kwargs or {}
    acqf_kwargs = acqf_kwargs or {}
    problem_kwargs = problem_kwargs or {}
    optimization_kwargs = optimization_kwargs or {}
    # TODO: use model list when there are constraints
    # or multiple objectives
    base_function = get_problem(name=function_name, dim=dim, **problem_kwargs)
    base_function.to(**tkwargs)
    binary_dims = base_function.integer_indices
    binary_mask = base_function.integer_bounds[1] - base_function.integer_bounds[0] == 1
    if binary_mask.any():
        binary_dims = (
            base_function.integer_indices.clone()
            .detach()
            .to(dtype=torch.int32)[binary_mask]
            .cpu()
            .tolist()
        )
    else:
        binary_dims = []

    is_constrained = isinstance(base_function, ConstrainedBaseTestProblem)
    num_constraints = base_function.num_constraints if is_constrained else 0

    is_moo = base_function.is_moo
    model_kwargs.setdefault("use_model_list", is_moo or is_constrained)
    kernel_type = model_kwargs.get("kernel_type")
    if "cont_optim__round_after" in label:
        if kernel_type in (
            "mixed_categorical",
            "mixed_latent",
        ):
            # cannot use a continuous relaxation + gradient optimization with mixed categorical
            model_kwargs["kernel_type"] = "mixed"
    elif "__ste__" in label:
        acqf_kwargs["use_ste"] = True
    if kernel_type in ("mixed_categorical", "mixed_latent"):
        acqf_kwargs["apply_numeric"] = True
    # set default optimization parameters
    optimization_kwargs.setdefault("num_restarts", 20)
    optimization_kwargs.setdefault("raw_samples", 1024)
    options = optimization_kwargs.get("options")
    if options is None:
        options = {}
        optimization_kwargs["options"] = options
    options.setdefault("batch_limit", 5)
    options.setdefault("init_batch_limit", 32)
    options.setdefault("maxiter", 200)
    if "__fin_diff__" in label:
        options["with_grad"] = False
    if options.get("sample_around_best", False):
        sigma = torch.full((base_function.dim,), 1e-3, **tkwargs)
        sigma[base_function.integer_indices] = 0.5 / (
            base_function.integer_bounds[1] - base_function.integer_bounds[0]
        )
        options["sample_around_best_sigma"] = sigma
        options["sample_around_best_subset_sigma"] = sigma

    exact_rounding_func = get_exact_rounding_func(
        bounds=base_function.one_hot_bounds,
        integer_indices=base_function.integer_indices.tolist(),
        categorical_features=base_function.categorical_features,
        initialization=False,
    )
    init_exact_rounding_func = get_exact_rounding_func(
        bounds=base_function.one_hot_bounds,
        integer_indices=base_function.integer_indices.tolist(),
        categorical_features=base_function.categorical_features,
        initialization=True,
    )
    standard_bounds = torch.ones(2, base_function.effective_dim, **tkwargs)
    standard_bounds[0] = 0
    # Get the initial data.
    if n_initial_points is None:
        n_initial_points = min(20, 2 * base_function.effective_dim)
    X, Y = generate_initial_data(
        n=n_initial_points,
        base_function=base_function,
        bounds=standard_bounds,
        tkwargs=tkwargs,
        init_exact_rounding_func=init_exact_rounding_func,
    )

    # Set some counters to keep track of things.
    start_time = time()
    existing_iterations = 0
    wall_time = torch.zeros(iterations, dtype=dtype)
    if is_moo:
        bd = DominatedPartitioning(ref_point=base_function.ref_point, Y=Y)
        # Abusing this variable name. This is HV.
        best_objs = bd.compute_hypervolume().view(-1).cpu()
    elif is_constrained:
        # compute feasibility
        feas = (Y[..., 1:] >= 0).all(dim=-1)
        if feas.any():
            best_objs = Y[feas, 0].max().view(-1).cpu()
        else:
            best_objs = torch.tensor([float("-inf")], dtype=Y.dtype)
    else:
        if base_function.objective_weights is not None:
            obj = Y @ base_function.objective_weights
        else:
            obj = Y
        best_objs = obj.max().view(-1).cpu()

    if use_trust_region:
        assert not is_moo
        trbo_state = TurboState(
            dim=base_function.effective_dim,
            batch_size=batch_size,
            is_constrained=is_constrained,
        )

    # setup nevergrad_portfolio
    if label == "nevergrad_portfolio":
        params = []
        for i in base_function.cont_indices:
            params.append(
                ng.p.Scalar(
                    lower=base_function.bounds[0, i].item(),
                    upper=base_function.bounds[1, i].item(),
                )
            )
        for i in base_function.integer_indices:
            params.append(
                ng.p.TransitionChoice(
                    list(
                        range(
                            int(base_function.bounds[0, i].item()),
                            int(base_function.bounds[1, i].item()) + 1,
                        )
                    )
                )
            )
        for i in base_function.categorical_indices:
            params.append(
                ng.p.Choice(
                    list(
                        range(
                            int(base_function.bounds[0, i].item()),
                            int(base_function.bounds[1, i].item()) + 1,
                        )
                    )
                )
            )
        params = ng.p.Instrumentation(*params)
        ohe_to_numeric = OneHotToNumeric(
            categorical_features=base_function.categorical_features,
            transform_on_train=True,
        )
        X_numeric = ohe_to_numeric(X)
        params.value = (tuple(X_numeric[-1].tolist()),{})
        optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(
            parametrization=params,
            budget=iterations + X.shape[0],
            num_workers=1,
        )
        optimizer.ask() # clear initial value
        if len(base_function.categorical_features) > 0:
            X_numeric[..., base_function.categorical_indices] = normalize(
                X_numeric[..., base_function.categorical_indices],
                base_function.categorical_bounds,
            )
        X_numeric = unnormalize(X_numeric, base_function.bounds)
        for xi, yi in zip(X_numeric.cpu().numpy(), Y.cpu().numpy()):
            xi = optimizer.parametrization.spawn_child(
                new_value=(tuple(xi.tolist()), {})
            )
            optimizer.tell(xi, yi.item())

    # BO loop for as many iterations as needed.
    for i in range(existing_iterations, iterations):
        print(
            f"Starting label {label}, seed {seed}, iteration {i}, "
            f"time: {time()-start_time}, current best obj: {best_objs[-1]}."
        )
        # Fit the model.
        mll, model, ecdfs = initialize_model(
            train_x=X,
            train_y=Y,
            binary_dims=binary_dims,
            categorical_features=base_function.categorical_features,
            function_name=function_name,
            **model_kwargs,
        )
        fit_gpytorch_model(mll)
        if label == "sobol":
            raw_candidates = (
                draw_sobol_samples(
                    bounds=standard_bounds,
                    n=1,
                    q=batch_size,
                )
                .squeeze(0)
                .to(**tkwargs)
            )
            candidates = init_exact_rounding_func(raw_candidates)
        elif label == "nevergrad_portfolio":
            X_numeric = ohe_to_numeric(X[-1:])
            if len(base_function.categorical_features) > 0:
                X_numeric[..., base_function.categorical_indices] = normalize(
                    X_numeric[..., base_function.categorical_indices],
                    base_function.categorical_bounds,
                )
            X_numeric = unnormalize(X_numeric, base_function.bounds)
            xi = optimizer.parametrization.spawn_child(
                new_value=(tuple(X_numeric.view(-1).tolist()), {})
            )
            optimizer.tell(xi, Y[-1].item())
            candidates_numeric = torch.tensor(
                optimizer.ask().value[0], dtype=X.dtype, device=X.device
            ).view(1, -1)
            candidates = normalize(candidates_numeric, base_function.bounds)
            if len(base_function.categorical_features) > 0:
                candidates[..., base_function.categorical_indices] = unnormalize(
                    candidates[..., base_function.categorical_indices],
                    base_function.categorical_bounds,
                )
            candidates = ohe_to_numeric.untransform(candidates)
        else:
            # Construct the acqf.
            acqf_exact_rounding_func = get_exact_rounding_func(
                bounds=base_function.one_hot_bounds,
                integer_indices=base_function.integer_indices.tolist(),
                categorical_features=base_function.categorical_features,
                initialization=False,
                return_numeric=acqf_kwargs.get("apply_numeric", False),
                use_ste=acqf_kwargs.get("use_ste", False),
            )
            Y_tf = (
                apply_normal_copula_transform(Y=Y, ecdfs=ecdfs)[0]
                if model_kwargs.get("copula", False)
                else Y
            )
            acq_func = get_acqf(
                label=label,
                mc_samples=mc_samples,
                model=model,
                X_baseline=X,
                num_constraints=num_constraints,
                iteration=i + 1,
                tkwargs=tkwargs,
                base_function=base_function,
                batch_size=batch_size,
                exact_rounding_func=acqf_exact_rounding_func,
                train_Y=Y_tf,
                **acqf_kwargs,
            )
            if "pr" in label:
                options["nonnegative"] = is_nonnegative(acq_func.acq_function)

            if use_trust_region:
                scaled_length = trbo_state.length * (
                    standard_bounds[1] - standard_bounds[0]
                )
                if is_constrained:
                    feas = (Y[..., 1:] >= 0).all(dim=-1)
                    if feas.any():
                        merit = Y[:, 0].clone()
                        merit[~feas] = -float("inf")
                        x_center = X[merit.argmax()]
                    else:
                        violation = torch.clamp_max(Y[..., 1:], 0.0).abs().sum(dim=-1)
                        x_center = X[violation.argmin()]
                else:
                    x_center = X[Y.argmax()].clone()
                bounds = torch.stack(
                    (x_center - scaled_length, x_center + scaled_length)
                )
                # Clamp bounds
                bounds[0] = torch.maximum(bounds[0], standard_bounds[0])
                bounds[1] = torch.minimum(bounds[1], standard_bounds[1])
                # Reset binary dimenions
                bounds[0, binary_dims] = 0
                bounds[1, binary_dims] = 1
                # Reset categorical_dims
                if len(base_function.categorical_features) > 0:
                    start = base_function.categorical_indices.min().item()
                    bounds[0, start:] = 0
                    bounds[1, start:] = 1
            else:
                bounds = standard_bounds

            # Optimize the acqf.
            torch.cuda.empty_cache()
            if "enumerate" in label:
                # enumerate the discrete options and optimize the continuous
                # parameters for each discrete option, if there are any

                # construct a list of dictionaries mapping indices in one-hot space
                # to parameter values.
                discrete_options = generate_discrete_options(
                    base_function=base_function,
                    return_tensor=base_function.cont_indices.shape[0] == 0,
                )
                if base_function.cont_indices.shape[0] > 0:
                    # optimize mixed
                    candidates, _ = optimize_acqf_mixed(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=batch_size,
                        fixed_features_list=discrete_options,
                        **optimization_kwargs,
                    )
                else:
                    # optimize discrete
                    candidates, _ = optimize_acqf_discrete(
                        acq_function=acq_func,
                        q=batch_size,
                        choices=discrete_options,
                        **optimization_kwargs,
                    )
            else:
                raw_candidates, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=bounds,
                    q=batch_size,
                    **optimization_kwargs,
                    # return candidates for all random restarts
                    return_best_only=False,
                )
                # round candidates
                candidates = exact_rounding_func(raw_candidates)
                # compute acquisition values of rounded candidates
                # and select best across restarts
                with torch.no_grad():
                    # TODO: support batches here
                    if batch_size > 1:
                        raise NotImplementedError
                    best_idx = acq_func(candidates).argmax().item()
                candidates = candidates[best_idx]

            torch.cuda.empty_cache()
            # free memory
            del acq_func, mll, model
            gc.collect()

        # Get the new observations and update the data.
        new_y = eval_problem(candidates, base_function=base_function)

        if use_trust_region:
            old_length = trbo_state.length
            trbo_state = update_state(state=trbo_state, Y_next=new_y)
            if trbo_state.length != old_length:
                print(
                    f"TR length changed from {old_length:.3f} to {trbo_state.length:3f}"
                )
            if trbo_state.restart_triggered:
                print("Restarting trust region")
                trbo_state = TurboState(
                    dim=base_function.effective_dim,
                    batch_size=batch_size,
                    is_constrained=is_constrained,
                )

        X = torch.cat([X, candidates], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        wall_time[i] = time() - start_time
        # TODO: add support for constraints by applying feasibility
        if is_moo:
            bd = DominatedPartitioning(ref_point=base_function.ref_point, Y=Y)
            # Abusing this variable name. This is HV.
            best_obj = bd.compute_hypervolume()
        elif is_constrained:
            # compute feasibility
            feas = (Y[..., 1:] >= 0).all(dim=-1)
            if feas.any():
                best_obj = Y[feas, 0].max()
            else:
                best_obj = torch.tensor([float("-inf")], dtype=Y.dtype, device=Y.device)
        else:
            if base_function.objective_weights is not None:
                obj = Y @ base_function.objective_weights
            else:
                obj = Y
            best_obj = obj.max().view(-1).cpu()
        best_objs = torch.cat([best_objs, best_obj.view(-1).cpu()], dim=0)

        # Periodically save the output.
        if save_frequency is not None and iterations % save_frequency == 0:
            output_dict = {
                "label": label,
                "X": X.cpu(),
                "Y": Y.cpu(),
                "wall_time": wall_time[: i + 1],
                "best_objs": best_objs,
            }
            save_callback(output_dict)

        # Save the final output
        output_dict = {
            "label": label,
            "X": X.cpu(),
            "Y": Y.cpu(),
            "wall_time": wall_time,
            "best_objs": best_objs,
        }
        save_callback(output_dict)
