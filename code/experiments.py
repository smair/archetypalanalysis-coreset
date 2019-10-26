#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ray
import numpy as np
from tqdm import tqdm
from time import time

from coresets import *
from archetypalanalysis import *


def experiment_AA_full(X, k):
    t_start = time()
    # initialize archetypes via FurthestSum
    ind = FurthestSum(X, k)
    Z_init = X[ind].copy()
    # run Archetypal Analysis
    Z, A, B, rss = ArchetypalAnalysis(X, Z_init, k)
    t_end = time()
    runtime = t_end - t_start
    print(len(rss))
    # recompute the load matrix on all data (just to be sure)
    A = ArchetypalAnalysis_compute_A(X, Z)
    # measure the error on all data
    rss = RSS_Z(X, A, Z)
    return rss, runtime


@ray.remote
def experiment_AA_uniform_sample(X, k, m, repetitions):
    n = X.shape[0]
    res = []
    res_time = []
    for i in range(repetitions):
        t_start = time()
        # obtain subset
        X_US = uniform_sample(X, m)
        # initialize archetypes via FurthestSum
        ind = FurthestSum(X_US, k)
        Z_init = X_US[ind].copy()
        # run Archetypal Analysis
        Z, A, B, rss = ArchetypalAnalysis(X_US, Z_init, k)
        t_end = time()
        runtime = t_end - t_start
        res_time.append(runtime)
        print(
            "{}iter, exp_AA_us, k={}, m={}, {}/{}".format(
                len(rss), k, m, i + 1, repetitions
            )
        )
        # recompute the load matrix on all data
        A = ArchetypalAnalysis_compute_A(X, Z)
        # measure the error on all data
        rss = RSS_Z(X, A, Z)
        res.append(rss)
    return res, res_time


@ray.remote
def experiment_AA_coreset(X, k, m, repetitions):
    res = []
    res_time = []
    for i in range(repetitions):
        t_start = time()
        # obtain subset
        X_C, w_C = coreset(X, m)
        # initialize archetypes via FurthestSum
        ind = FurthestSum(X_C, k)
        Z_init = X_C[ind].copy()
        # run weighted Archetypal Analysis
        W = np.diag(np.sqrt(w_C))
        Z, A, B, rss = weightedArchetypalAnalysis(X_C, Z_init, k, W)
        t_end = time()
        runtime = t_end - t_start
        res_time.append(runtime)
        print(
            "{}iter, exp_AA_cs, k={}, m={}, {}/{}".format(
                len(rss), k, m, i + 1, repetitions
            )
        )
        # recompute the load matrix on all data
        A = ArchetypalAnalysis_compute_A(X, Z)
        # measure the error on all data
        rss = RSS_Z(X, A, Z)
        res.append(rss)
    return res, res_time


@ray.remote
def experiment_AA_lightweight_coreset(X, k, m, repetitions):
    res = []
    res_time = []
    for i in range(repetitions):
        t_start = time()
        # obtain subset
        X_C, w_C = lightweight_coreset(X, m)
        # initialize archetypes via FurthestSum
        ind = FurthestSum(X_C, k)
        Z_init = X_C[ind].copy()
        # run weighted Archetypal Analysis
        W = np.diag(np.sqrt(w_C))
        Z, A, B, rss = weightedArchetypalAnalysis(X_C, Z_init, k, W)
        t_end = time()
        runtime = t_end - t_start
        res_time.append(runtime)
        print(
            "{}iter, exp_AA_lwcs, k={}, m={}, {}/{}".format(
                len(rss), k, m, i + 1, repetitions
            )
        )
        # recompute the load matrix on all data
        A = ArchetypalAnalysis_compute_A(X, Z)
        # measure the error on all data
        rss = RSS_Z(X, A, Z)
        res.append(rss)
    return res, res_time


@ray.remote
def experiment_AA_lucic_coreset(X, k, m, repetitions):
    res = []
    res_time = []
    for i in range(repetitions):
        t_start = time()
        # obtain subset
        X_C, w_C = lucic_coreset(X, m, k)
        # initialize archetypes via FurthestSum
        ind = FurthestSum(X_C, k)
        Z_init = X_C[ind].copy()
        # run weighted Archetypal Analysis
        W = np.diag(np.sqrt(w_C))
        Z, A, B, rss = weightedArchetypalAnalysis(X_C, Z_init, k, W)
        t_end = time()
        runtime = t_end - t_start
        res_time.append(runtime)
        print(
            "{}iter, exp_AA_lucic, k={}, m={}, {}/{}".format(
                len(rss), k, m, i + 1, repetitions
            )
        )
        # recompute the load matrix on all data
        A = ArchetypalAnalysis_compute_A(X, Z)
        # measure the error on all data
        rss = RSS_Z(X, A, Z)
        res.append(rss)
    return res, res_time


def experiment_AA_uniform_sample_parallel(X, k, m, repetitions, parallel=10):
    reps = int(repetitions / parallel)

    # start $parallel tasks in parallel
    result_ids = []
    for i in range(parallel):
        result_ids.append(experiment_AA_uniform_sample.remote(X, k, m, reps))

    result = ray.get(result_ids)

    res = np.array(list(map(lambda x: x[0], result))).flatten()
    res_time = np.array(list(map(lambda x: x[1], result))).flatten()

    return res, res_time


def experiment_AA_coreset_parallel(X, k, m, repetitions, parallel=10):
    reps = int(repetitions / parallel)

    # start $parallel tasks in parallel
    result_ids = []
    for i in range(parallel):
        result_ids.append(experiment_AA_coreset.remote(X, k, m, reps))

    result = ray.get(result_ids)

    res = np.array(list(map(lambda x: x[0], result))).flatten()
    res_time = np.array(list(map(lambda x: x[1], result))).flatten()

    return res, res_time


def experiment_AA_lightweight_coreset_parallel(X, k, m, repetitions, parallel=10):
    reps = int(repetitions / parallel)

    # start $parallel tasks in parallel
    result_ids = []
    for i in range(parallel):
        result_ids.append(experiment_AA_lightweight_coreset.remote(X, k, m, reps))

    result = ray.get(result_ids)

    res = np.array(list(map(lambda x: x[0], result))).flatten()
    res_time = np.array(list(map(lambda x: x[1], result))).flatten()

    return res, res_time


def experiment_AA_lucic_coreset_parallel(X, k, m, repetitions, parallel=10):
    reps = int(repetitions / parallel)

    # start $parallel tasks in parallel
    result_ids = []
    for i in range(parallel):
        result_ids.append(experiment_AA_lucic_coreset.remote(X, k, m, reps))

    result = ray.get(result_ids)

    res = np.array(list(map(lambda x: x[0], result))).flatten()
    res_time = np.array(list(map(lambda x: x[1], result))).flatten()

    return res, res_time
