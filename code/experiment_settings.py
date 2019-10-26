#!/usr/bin/env python3
# -*- coding: utf-8 -*-

data_path = "/home/sebastian/data/"
results_path = "/home/sebastian/test/run/"

# specify list of sample sizes m
M = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

# number of repetitions per experiment
repetitions = 50

# renaming some data sets
data_name = { # old:new
    "ijcnn1": "Ijcnn1",
    "pose": "Pose",
    "song": "Song",
    "covertype": "Covertype",
}
