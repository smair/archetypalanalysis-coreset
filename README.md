# Coresets for Archetypal Analysis

This repository contains the source code of the paper **Coresets for Archetypal Analysis**.

## Abstract
Archetypal analysis (AA) represents instances as linear mixtures of prototypes (the archetypes) that lie on the boundary of the convex hull of the data. Archetypes are thus  often better interpretable than factors computed by other matrix factorization techniques. However, the interpretability comes with high computational cost due to additional convexity-preserving constraints. In this paper, we propose efficient coresets for archetypal analysis. Theoretical guarantees are derived by showing that quantization errors of k-means upper bound archetypal analysis; the computation of a provable absolute-coreset can be performed in only two passes over the data. Empirically, we show that the coresets lead to improved performance on several data sets.

<p align="center">
  <img src="main.png" alt="visualization of the approach"/>
</p>


## Prerequisites
You might consider building the nnls module which has a higher number of max. iterations for improved stability of solving non-negative least squares problems which is needed for Archetypal Analysis.
```bash
$ bash build_nnls.sh
```

The code was tested with the following versions:

* python 3.7.3
* numpy 1.16.4
* scipy 1.3.1
* sklearn 0.21.2
* ray 0.7.3

In the paper we used the following four datasets which are not included in this repository:

* [**Ijcnn1**](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)
* [**Pose**](http://vision.imar.ro/human3.6m/challenge_open.php)
* [**Song**](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)
* [**Covertype**](https://archive.ics.uci.edu/ml/datasets/covertype)

You have to download them yourself and specify the location within experiment_settings.py.

## Usage

The file example.py shows how to run Archetypal Analysis on the full data set as well as on a coreset and compares the results.

To perform experiments similar to those in the paper you can run
```bash
$ python3 run_experiment.py NAME_OF_DATASET
```

To replicate the experiments of the paper you have to run
```bash
$ python3 run_experiment.py ijcnn1
$ python3 run_experiment.py pose
$ python3 run_experiment.py song
$ python3 run_experiment.py covertype
```
