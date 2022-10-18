#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import pickle
import random
import sys
from itertools import product
from logging import Logger
from typing import Optional, Union

import numpy as np
import torch
from botorch.test_functions.base import MultiObjectiveTestProblem

from discrete_mixed_bo.problems.base import DiscreteTestProblem


class NASHPOBench2API:
    def __init__(
        self,
        data: dict,
        seed: int = 0,
        logger: Optional[Logger] = None,
        verbose: bool = False,
    ):
        self.logger = logger
        self.verbose = verbose
        if logger is None:
            self.logger = logging.getLogger(__name__)
            handler = logging.StreamHandler()
            self.logger.addHandler(handler)
            self.logger.propagate = False
        self.bench12 = data["bench12"]
        self.cellinfo = data["cellinfo"]
        self.avgaccs200 = data["avgaccs200"]
        # set serach space
        self.cellcodes = sorted(list(set(self.cellinfo["hash"].keys())))
        self.lrs = sorted(list(set(self.bench12["lr"].values())))
        self.batch_sizes = sorted(list(set(self.bench12["batch_size"].values())))
        self.seeds = sorted(list(set(self.bench12["seed"].values())))
        self.epochs = [12, 200]
        # set objects for log
        self._init_logdata()
        # set seed
        if seed:
            self.set_seed(seed)
        else:
            self.seed = self.seeds[0]

    def _init_logdata(self):
        ## acc
        self.acc_trans = []
        self.best_acc = -1
        self.best_acc_trans = []
        ## cost
        self.total_cost = 0.0
        self.total_cost_trans = []
        self.best_cost = None
        ## key
        self.key_trans = []
        self.best_key = None
        self.best_key_trans = []
        # set verbose
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        return

    def __len__(self):
        return len(self.cellcodes) * len(self.lrs) * len(self.batch_sizes)

    def __str__(self):
        return "NAS-HPO-Bench-II"

    def get_key_from_idx(self, idx: int):
        key = {
            "cellcode": self.cellcodes[
                int(idx / (len(self.lrs) * len(self.batch_sizes)))
            ],
            "lr": self.lrs[int(idx / len(self.batch_sizes)) % len(self.lrs)],
            "batch_size": self.batch_sizes[idx % len(self.batch_sizes)],
        }
        return key

    def get_idx_from_key(self, key: dict):
        cellidx = self.cellcodes.index(key["cellcode"])
        lridx = self.lrs.index(key["lr"])
        batchidx = self.batch_sizes.index(key["batch_size"])
        return (
            cellidx * len(self.lrs) * len(self.batch_sizes)
            + lridx * len(self.batch_sizes)
            + batchidx
        )

    def __getitem__(
        self,
        idx: int,
    ):
        key = self.get_key_from_idx(idx)
        return self.query_by_key(**key)

    def query_by_index(
        self,
        cell_idx: int,
        lr_idx: int,
        batch_size_idx: int,
        epoch: Union[int, str] = 12,
        iepoch: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        cellcode = self.cellcodes[cell_idx]
        lr = self.lrs[lr_idx]
        batch_size = self.batch_sizes[batch_size_idx]
        if seed:
            self.set_seed(seed)
        return self.query_by_key(cellcode, lr, batch_size, epoch=epoch, iepoch=iepoch)

    def query_by_key(
        self,
        cellcode: str,
        lr: float,
        batch_size: int,
        epoch: Union[int, str] = 12,
        mode: str = "valid",
        iepoch: Optional[int] = None,
        seed: Optional[int] = None,
        enable_log: bool = True,
    ):
        # check if a key is valid
        self._check_key(cellcode, lr, batch_size, epoch)
        assert mode in ["train", "valid", "test"], ValueError(
            f"mode {mode} should be train, valid, or test"
        )
        # iepoch
        if epoch != 12:
            assert iepoch is None, ValueError(
                f"iepoch is not available in epoch {epoch}"
            )
        if iepoch == None:
            iepoch = epoch
        assert iepoch <= epoch, ValueError(
            f"iepoch {iepoch} is graeter than epoch {epoch}"
        )
        # set seed
        if seed:
            self.set_seed(seed)
        # cellcode to hashvalue
        hashv = self._cellcode2hash(cellcode)

        # 12 epoch
        if epoch == 12:
            # acc
            bench = self.bench12
            acc = bench[f"{mode}_acc_{iepoch-1}"][(hashv, lr, batch_size, self.seed)]
            # cost
            ## if iepoch == 12 (, the cost is pre-calculated)
            if iepoch == epoch:
                if mode == "train":
                    cost = bench[f"total_train_time"][
                        (hashv, lr, batch_size, self.seed)
                    ]
                elif mode == "valid":
                    cost = bench[f"total_trainval_time"][
                        (hashv, lr, batch_size, self.seed)
                    ]
                elif mode == "test":
                    cost = (
                        bench[f"total_trainval_time"][
                            (hashv, lr, batch_size, self.seed)
                        ]
                        + bench[f"total_test_time"][(hashv, lr, batch_size, self.seed)]
                    )
            ## else (less than 12 epoch)
            else:
                if mode == "train":
                    time_modes = ["train"]
                elif mode == "valid":
                    time_modes = ["train", "valid"]
                elif mode == "test":
                    time_modes = ["train", "valid", "test"]
                tmp = [
                    bench[f"{m}_time_{i}"][(hashv, lr, batch_size, self.seed)]
                    for m, i in product(time_modes, range(iepoch))
                ]
                cost = sum(tmp)

            key = {"cellcode": cellcode, "lr": lr, "batch_size": batch_size}
            if enable_log:
                self._write_log(acc, cost, key)
            return acc, cost

        # 200 epoch
        elif epoch == 200:
            # the expected value of test accuracy
            bench = self.avgaccs200
            acc = bench["avg_acc"][(hashv, lr, batch_size)]
            return acc, None

    def _write_log(
        self,
        acc: float,
        cost: float,
        key: dict,
    ):
        if len(self.acc_trans) == 0:
            self.logger.debug(
                f'   {"valid acc":<8}   {"cost":<8}    {"cellcode"} {"lr":<7} {"batch_size":<3}'
            )
        self.logger.debug(
            f'{acc:>8.2f} %   {cost:>8.2f} sec  {key["cellcode"]} {key["lr"]:<7.5f} {key["batch_size"]:<3}'
        )
        # current status
        self.acc_trans.append(acc)
        self.key_trans.append(key)
        self.total_cost += cost
        self.total_cost_trans.append(self.total_cost)
        # update the best status
        if self.best_key is None or self.best_acc < acc:
            self.best_acc, self.best_cost, self.best_key = acc, cost, key
        # current best status
        self.best_acc_trans.append(self.best_acc)
        self.best_key_trans.append(self.best_key)
        return

    def get_total_cost(self):
        return self.total_cost

    def get_results(self, epoch: Union[int, str] = "both", mode: str = "test"):
        # log
        self.logger.info("-" * 23 + " finished " + "-" * 23)
        self.logger.info("The best setting is")
        self.logger.info(
            f'   {"valid acc":<8}   {"cost":<8}    {"cellcode"} {"lr":<7} {"batch_size":<3}'
        )
        self.logger.info(
            f'{self.best_acc:>8.2f} %   {self.best_cost:>8.2f} sec  {self.best_key["cellcode"]} {self.best_key["lr"]:<7.5f} {self.best_key["batch_size"]:<3}'
        )
        self.logger.info(
            f" in {len(self.key_trans)} trials ({self.total_cost:.2f} sec)"
        )

        # get the test accuracies of the best-valid-acc model (finalaccs)
        if epoch == "both":
            epochs = [12, 200]
        else:
            epochs = [epoch]
        self.logger.info("-" * 56)
        self.final_accs = []
        for e in epochs:
            final_acc, _ = self.query_by_key(
                **self.best_key, epoch=e, mode=mode, enable_log=False
            )
            self.final_accs.append(final_acc)
            self.logger.info(f"{e}-epoch {mode} accuracy is {final_acc:.2f}%")
        self.logger.info("-" * 56)
        # return results
        nlist = [
            "acc_trans",
            "key_trans",
            "best_acc_trans",
            "best_key_trans",
            "total_cost_trans",
            "final_accs",
        ]
        return {n: eval("self." + n, {"self": self}) for n in nlist}

    def reset_logdata(self, logger: Optional[Logger] = None, verbose: bool = None):
        if logger is not None:
            self.logger = logger
        if verbose is not None:
            self.verbose = verbose
        self._init_logdata()
        return

    def _cellcode2hash(self, cellcode: str):
        return self.cellinfo["hash"][cellcode]

    def get_random_key(self):
        cellcode = random.choice(self.cellcodes)
        lr = random.choice(self.lrs)
        batch_size = random.choice(self.batch_sizes)
        return {"cellcode": cellcode, "lr": lr, "batch_size": batch_size}

    def _check_key(self, cellcode: str, lr: float, batch_size: int, epoch: int):
        if cellcode not in self.cellcodes:
            raise ValueError(f"choose a cellcode {cellcode} from search space.")
        if lr not in self.lrs:
            raise ValueError(f"choose lr from {self.lrs}.")
        if batch_size not in self.batch_sizes:
            raise ValueError(f"choose batch size from {self.batch_sizes}.")
        if epoch not in self.epochs:
            raise ValueError(f"choose epoch from {self.epochs}.")
        return

    def get_search_space(self):
        return self.cellcodes, self.lrs, self.batch_sizes

    def set_seed(self, seed: int):
        if seed in self.seeds:
            self.seed = seed
        else:
            raise ValueError(f"choose a seed value from {self.seeds}.")


# class NASHPOBenchII(MultiObjectiveTestProblem, DiscreteTestProblem):
class NASHPOBenchII(DiscreteTestProblem):
    """
    NAS-HPO-Bench-II dataset that jointly optimizes the architecture and hyperparameters on a simplified NAS-Bench-201
        search space.
    """

    _bounds = [(0, 3) for _ in range(6)] + [(0, 5), (0, 7)]
    # first 6: NAS dimensions -- specify the operators on the cell
    # final 2: batch sizes -- log-ordinal variables
    _ref_point = [
        0.0933,
        0.0144,
    ]  # <- found by running this python file as main using botorch heuristic in finding ref point
    # (see the main code block to see how this is determined)
    dim: int = 8

    def __init__(
        self,
        data: dict,
        nb201_path: str = None,
        num_objectives: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = True,
        use_12_epoch_result: bool = False,
        use_log: bool = False,
    ):
        """
        Args:
            data_path: path to the stored NAS-HPO-Bench-II dataset
            nb201_path: Optional. Compulsory if num_objectives > 1. Path to stored NATS-Bench dataset
            num_objectives: 1 or 2.
                1: only validation error is returned
                2: (validation error, latency) --> both need to be minimized
            noise_std:
            negate:
            use_12_epoch_result: if True, return the 12-epoch result (otherwise return 200-epoch)
        """
        super(NASHPOBenchII, self).__init__(
            negate=negate,
            noise_std=noise_std,
            categorical_indices=list(range(6)),
            integer_indices=[6, 7],
        )
        assert num_objectives in [1, 2]
        if num_objectives > 1:
            raise NotImplementedError
            # assert nb201_path is not None, 'NB201 path is needed in multi-objective mode!'
        self.num_objectives = num_objectives
        self.use_12_epoch_result = use_12_epoch_result
        self.api = NASHPOBench2API(data=data)
        # self.nb201_api = nats_bench.create(nb201_path, 'tss', fast_mode=True, verbose=True)
        self.nb201_api = None
        # note this is specified by the NAS-HPO-Bench-II authors
        self.primitives = {
            0: "nor_conv_3x3",
            1: "avg_pool_3x3",
            2: "skip_connect",
            3: "none",
        }
        self.use_log = use_log

    def parse_input(self, x_array):
        """Parse an array input into a format understood by the NAS-HPO-Bench-II API"""
        assert x_array.shape[0] == self.dim
        x_array = x_array.int().detach().cpu().numpy()
        nas_code, bs_code, lr_code = x_array[:6], x_array[6], x_array[7]
        cellcode = f"{nas_code[0]}|{nas_code[1]}{nas_code[2]}|{nas_code[3]}{nas_code[4]}{nas_code[5]}"
        bs = int(2 ** (bs_code + 4))  # so 0 -> 16, 1 -> 32, ..., 5 -> 512
        lr = float(
            0.003125 * (2**lr_code)
        )  # so 0 -> 0.003125, 1 -> 0.00625, ..., 7 -> 0.4
        return cellcode, bs, lr

    def get_nb201_code(self, x_array):
        """Get a NAS-Bench-201/NATS-Bench style code -- used to query the model parameters & FLOPS"""
        assert x_array.shape[0] == self.dim
        nas_code = x_array[:6]
        op_list = [self.primitives[i] for i in nas_code]
        arch_str = get_string_from_op_list(op_list)
        return arch_str

    def nashpo2nb201(self, nas_code):
        """Convert a NAS-HPO-Bench-II code back to NATSBench/NAS-Bench-201 code"""
        nas_code_splitted = nas_code.split("|")
        op_list = []
        for node in nas_code_splitted:  # e.g. ['1', '12', '123']
            for prim in node:
                op_list.append(self.primitives[int(prim)])
        arch_str = get_string_from_op_list(op_list)
        return arch_str

    def _evaluate_single(self, input: torch.Tensor):
        cellcode, bs, lr = self.parse_input(input)

        accs, costs = [], []
        for seed in self.api.seeds:
            acc, cost = self.api.query_by_key(
                cellcode=cellcode,
                batch_size=bs,
                lr=lr,
                epoch=12 if self.use_12_epoch_result else 200,
                seed=seed,
            )
            accs.append(acc)
            costs.append(cost)
        print(accs)
        acc = sum(accs) / len(accs)
        if costs[0] is not None:
            cost = sum(costs) / len(costs)
        err = (100.0 - acc) / 100.0
        if self.num_objectives == 1:
            return torch.tensor(err, dtype=torch.float)
        else:  # also query the model size
            nb201_str = self.get_nb201_code(input)
            arch_index = self.nb201_api.query_index_by_arch(nb201_str)
            cost_result = self.nb201_api.get_cost_info(arch_index, "cifar10")
            latency = cost_result["latency"]
            return torch.tensor([err, latency], dtype=torch.float).view(-1)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        res = (
            torch.stack(
                [self._evaluate_single(x) for x in X.cpu().view(-1, self.dim)],
            )
            .to(X)
            .view(*X.shape[:-1], self.num_objectives)
        )
        if self.use_log:
            res = res.log()
        if self.num_objectives == 1:
            return res.squeeze(-1)
        return res
