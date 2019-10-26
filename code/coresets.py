#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# "uniform" in the paper
def uniform_sample(X, m):
    n = X.shape[0]
    ind = np.random.choice(n, m)
    X_C = X[ind]
    return X_C


# "lw-cs" in the paper; outlined in Algorithm 1
def lightweight_coreset(X, m):
    # Scalable k-means clustering via lightweight coresets
    # Bachem et al. (2018)
    n = X.shape[0]
    dist = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
    q = 0.5 * 1 / n + 0.5 * dist / dist.sum()
    ind = np.random.choice(n, m, p=q)
    X_C = X[ind]
    w_C = 1 / (m * q[ind])
    return X_C, w_C


# Mahalanobis D^2-sampling
# needed by lucic_coreset()
def mahanalobis_d2_sampling(X, k):
    n = X.shape[0]
    i = np.random.choice(n, 1)
    B = X[i]
    for _ in range(k - 1):
        # d_A(x,y) = is \|x-y\|_2^2
        dist = np.array(list(map(lambda b: np.sum((X - b) ** 2, axis=1), B)))
        closest_cluster_id = dist.argmin(0)
        dist = dist[closest_cluster_id, np.arange(n)]
        p = dist / dist.sum()
        i = np.random.choice(n, 1, replace=False, p=p)[0]
        B = np.vstack((B, X[i]))
    return B


# "lucic-cs" in the paper
def lucic_coreset(X, m, k):
    # Strong Coresets for Hard and Soft Bregman Clustering with Applications to Exponential Family Mixtures
    # Lucic et al. (2016)
    n = X.shape[0]
    B = mahanalobis_d2_sampling(X, k)
    a = 16 * (np.log(k) + 2)
    # d_A(x,y) = is \|x-y\|_2^2
    dist = np.array(list(map(lambda b: np.sum((X - b) ** 2, axis=1), B)))
    closest_cluster_id = dist.argmin(0)
    dist = dist[closest_cluster_id, np.arange(n)]
    c = dist.mean()
    s = np.zeros(n)
    for i in range(n):
        Bi_cardinality = np.sum(closest_cluster_id == closest_cluster_id[i])
        tmp = dist[closest_cluster_id == closest_cluster_id[i]].sum()
        s[i] = (
            a * dist[i] / c
            + 2 * a * tmp / (Bi_cardinality * c)
            + 4 * n / Bi_cardinality
        )
    p = s / s.sum()
    ind = np.random.choice(n, m, p=p)
    X_C = X[ind]
    w_C = 1 / (m * p[ind])
    return X_C, w_C


# proposed coreset
# "abs-cs" in the paper; outlined in Algorithm 2
def coreset(X, m):
    n = X.shape[0]
    dist = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
    q = dist / dist.sum()
    ind = np.random.choice(n, m, p=q)
    X_C = X[ind]
    w_C = 1 / (m * q[ind])
    return X_C, w_C
