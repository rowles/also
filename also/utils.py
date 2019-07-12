import numpy as np


def partition_mat(X, n):
    cols_idx = [i for i in range(X.shape[1]) if i != n]
    x = X[:, cols_idx].copy()
    y = X[:, n].copy()
    return x, y


def weight_scores(y, scores, err):
    sq_dev = (y - np.mean(y)) ** 2
    rk = np.sqrt(err / np.sum(sq_dev))
    w = 1 - min(1, rk)
    w_scores = w * scores
    return w, w_scores


def score_instance(weights, w_sum):
    return np.sqrt(np.sum(weights) / w_sum)
