[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization

This is the code associated with the paper "[Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization](https://github.com/facebookresearch/bo_pr/blob/main/BO_Probabilistic_Reparameterization.pdf)."

Please cite our work if you find it useful.

    
    @inproceedings{daulton2022pr,
          title={Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization}, 
          author={Samuel Daulton and Xingchen Wan and and David Eriksson and Maximilian Balandat and Michael A. Osborne and Eytan Bakshy},
          booktitle={Advances in Neural Information Processing Systems 35},
          year={2022}
    }


## Getting started

From the base `bo_pr` directory run:

`pip install -e .`

## Structure

The code is structured in three parts.
- The utilities for constructing the acquisition functions and other helper methods are defined in `discrete_mixed_bo/`.
- The experiments are found in and ran from within `experiments/`. The `main.py` is used to run the experiments, and the experiment configurations are found in the `config.json` file of each sub-directory.

The individual experiment outputs were left out to avoid inflating the file size.

## Running Experiments

To run a basic benchmark based on the `config.json` file in `experiments/<experiment_name>` using `<algorithm>`:

```
cd experiments
python main.py <experiment_name> <algorithm> <seed>
```

The code refers to the algorithms using the following labels:
```
algorithms = [
    ("sobol", "Sobol"),
    ("cont_optim__round_after__ei", "Cont. Relax."),
    ("pr__ei", "PR"),
    ("exact_round__fin_diff__ei", "Exact Round"),
    ("exact_round__ste__ei", "Exact Round + STE"),
    ("cont_optim__round_after__ts", "Cont. Relax. + TS"),
    ("pr__ts", "PR + TS"),
    ("exact_round__fin_diff__ts", "Exact Round + TS"),
    ("exact_round__ste__ts", "Exact Round + TS + STE"),
    ("cont_optim__round_after__ucb", "Cont. Relax. + UCB"
    ("pr__ucb", "PR + UCB"),
    ("exact_round__fin_diff__ucb", "Exact Round + UCB"),
    ("exact_round__ste__ucb", "Exact Round + UCB + STE"),
    ("cont_optim__round_after__ehvi", "Cont. Relax."),
    ("pr__ehvi", "PR"),
    ("exact_round__fin_diff__ehvi", "Exact Round"),
    ("exact_round__ste__ehvi", "Exact Round + STE"),
    ("cont_optim__round_after__nehvi-1", ""Cont. Relax. + TS"),
    ("pr__nehvi-1", "PR + TS"),
    ("exact_round__fin_diff__nehvi-1", "Exact Round + TS"),
    ("exact_round__ste__nehvi-1","Exact Round + TS + STE"),
]
```

These algorithms can be modified by additional arguments in the `config.json` file such as `use_trust_region` and a "acqf_kwargs" dictionary containing the pair`"pr_use_analytic": true`.

Each folder under `experiments/` corresponds to the experiments in the paper according to the following mapping:
```
experiments = {
    "ackley13": "Ackley",
    "cco": "Cellular Network",
    "chemistry": "Chemical Reaction",
    "mixed_int_f1": "Mixed Int F1",
    "mixed_oil": "Oil Sorbent",
    "rosenbrock10_scaled": "Rosenbrock",
    "svm": "SVM",
    "welded_beam": "Welded Beam",
}
```
Additional folder provide the configurations for the trust region variants (simply adding `"use_trust_region": true`) and analytic PR where feasible simply adding (`"acqf_kwargs": {"pr_use_analytic": true}`). Note: this code can heavily exploit a GPU if available. 

## License
This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
