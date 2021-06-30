import numpy as np

import also


def test_partition_mat():
    mat = np.array([[1, 2, 3], [4, 5, 6]])

    x, y = also._partition_mat(mat, 0)

    np.testing.assert_array_equal(y, [1, 4])
    np.testing.assert_array_equal(x, [[2, 3], [5, 6]])

    x, y = also._partition_mat(mat, 1)

    np.testing.assert_array_equal(y, [2, 5])
    np.testing.assert_array_equal(x, [[1, 3], [4, 6]])

    x, y = also._partition_mat(mat, 2)

    np.testing.assert_array_equal(y, [3, 6])
    np.testing.assert_array_equal(x, [[1, 2], [4, 5]])


def test_weight_scores():
    y = np.array([1, 2, 3])
    scores = np.array([0.1, 0.1, 0.3])
    err = 0.1

    w, w_scores = also._weight_scores(y, scores, err)

    np.testing.assert_almost_equal(w, 0.7763, decimal=4)
    np.testing.assert_array_almost_equal(
        w_scores,
        [0.077, 0.077, 0.232],
        decimal=3
    )


def test_score_instance():
    weights = np.array([0.1, 0.3, 0.4])
    w_sum = 2

    scores = also._score_instance(weights, w_sum)
    np.testing.assert_almost_equal(
        scores,
        0.632,
        decimal=3
    )

