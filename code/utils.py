#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

from experiment_settings import *


def load_data(dataset, standardize=False):
    X = []
    y = []

    if dataset == "covertype":  # (581012, 54)
        # Forest cover type
        # https://archive.ics.uci.edu/ml/datasets/covertype
        X, y = load_svmlight_file(data_path + "covtype.libsvm.binary")
        X = np.asarray(X.todense())
    elif dataset == "ijcnn1":  # (49990, 22)
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        X, y = load_svmlight_file(data_path + "ijcnn1/ijcnn1")
        X = np.asarray(X.todense())
    elif dataset == "song":  # (515345, 90)
        # YearPredictionMSD is a subset of the Million Song Dataset
        # https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
        data = np.loadtxt(
            data_path + "YearPredictionMSD.txt", skiprows=0, delimiter=","
        )
        X = data[:, 1:]
        y = data[:, 0]
    elif dataset == "pose":  # (35832, 48)
        # ECCV 2018 PoseTrack Challenge
        # http://vision.imar.ro/human3.6m/challenge_open.php
        X = []
        for i in tqdm(range(1, 35832 + 1), desc="loading pose"):
            f = data_path + "Human3.6M/ECCV18_Challenge/Train/POSE/{:05d}.csv".format(i)
            data = np.loadtxt(f, skiprows=0, delimiter=",")
            X.append(data[1:, :].flatten())
        X = np.array(X)
    else:
        raise NotImplementedError

    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, y
