#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import *
from coresets import *
from archetypalanalysis import *


# set seed
np.random.seed(42)
# load data
X, y = load_data("ijcnn1")
# specify number of archetypes
k = 25
# specify coreset size
m = 1000


# initialize archetypes via FurthestSum
ind = FurthestSum(X, k)
Z_init = X[ind].copy()
# run Archetypal Analysis
Z, A, B, rss = ArchetypalAnalysis(X, Z_init, k)
# measure the error on all data
rss_full = RSS_Z(X, A, Z)


# compute proposed coreset
X_C, w_C = coreset(X, m)
# initialize archetypes via FurthestSum
ind = FurthestSum(X_C, k)
Z_init = X_C[ind].copy()
# run weighted Archetypal Analysis
W = np.diag(np.sqrt(w_C))
Z, A, B, rss = weightedArchetypalAnalysis(X_C, Z_init, k, W)
# recompute the load matrix on all data
A = ArchetypalAnalysis_compute_A(X, Z)
# measure the error on all data
rss_coreset = RSS_Z(X, A, Z)


# show relative error
rel_error = np.abs(rss_full - rss_coreset) / rss_full
print("relative error is {}".format(rel_error))
