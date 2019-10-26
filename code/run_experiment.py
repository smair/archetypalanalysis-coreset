#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import ray

from utils import *
from coresets import *
from experiments import *
from experiment_settings import *


ray.init()

dataset = str(sys.argv[1])
X, y = load_data(dataset)  # y won't be used

np.random.seed(0)

for k in [25, 100]:
    # k is the number of archetypes

    # Archetypal Analysis on all data
    print("Archetypal Analysis on all data")
    rss_full, time_full = experiment_AA_full(X, k)

    # Archetypal Analysis on uniform sample
    print("Archetypal Analysis on uniform sample")
    rss_uniform_sample = []
    time_uniform_sample = []
    for m in M:
        res, res_time = experiment_AA_uniform_sample_parallel(X, k, m, repetitions)
        rss_uniform_sample.append(res)
        time_uniform_sample.append(res_time)
    rss_uniform_sample = np.array(rss_uniform_sample)
    time_uniform_sample = np.array(time_uniform_sample)

    # Archetypal Analysis on lightweight coreset
    print("AA on lightweight coreset")
    rss_lw_coreset = []
    time_lw_coreset = []
    for m in M:
        res, res_time = experiment_AA_lightweight_coreset_parallel(X, k, m, repetitions)
        rss_lw_coreset.append(res)
        time_lw_coreset.append(res_time)
    rss_lw_coreset = np.array(rss_lw_coreset)
    time_lw_coreset = np.array(time_lw_coreset)

    # Archetypal Analysis on proposed coreset
    print("Archetypal Analysis on proposed coreset")
    rss_coreset = []
    time_coreset = []
    for m in M:
        res, res_time = experiment_AA_coreset_parallel(X, k, m, repetitions)
        rss_coreset.append(res)
        time_coreset.append(res_time)
    rss_coreset = np.array(rss_coreset)
    time_coreset = np.array(time_coreset)

    # Archetypal Analysis on lucic coreset
    print("Archetypal Analysis on lucic coreset")
    rss_lucic_coreset = []
    time_lucic_coreset = []
    for m in M:
        res, res_time = experiment_AA_lucic_coreset_parallel(X, k, m, repetitions)
        rss_lucic_coreset.append(res)
        time_lucic_coreset.append(res_time)
    rss_lucic_coreset = np.array(rss_lucic_coreset)
    time_lucic_coreset = np.array(time_lucic_coreset)

    # save results in npz file
    np.savez(
        results_path + dataset + "_coreset_k{}.npz".format(k),
        dataset=dataset,
        X=X,
        k=k,
        repetitions=repetitions,
        M=M,
        rss_full=rss_full,
        time_full=time_full,
        rss_lucic_coreset=rss_lucic_coreset,
        time_lucic_coreset=time_lucic_coreset,
        rss_uniform_sample=rss_uniform_sample,
        time_uniform_sample=time_uniform_sample,
        rss_lw_coreset=rss_lw_coreset,
        time_lw_coreset=time_lw_coreset,
        rss_coreset=rss_coreset,
        time_coreset=time_coreset,
    )
