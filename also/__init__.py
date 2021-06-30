from functools import partial

import numpy as np
import numpy.typing as npt

from sklearn.model_selection import KFold


def _partition_mat(X: npt.NDArray[np.float64], n: int) -> (npt.NDArray[np.float64], npt.NDArray[np.float64]):
    """ Select a column to take out of matrix
    """
    cols_idx = [i for i in range(X.shape[1]) if i != n]
    x = X[:, cols_idx]
    y = X[:, n]
    return x, y


def _weight_scores(y: npt.NDArray[np.float64], scores: npt.NDArray[np.float64], err: npt.NDArray[np.float64]) -> (float, npt.NDArray[np.float64]):
    """ Weight the predictions by the deviation
    """
    sq_dev = (y - np.mean(y)) ** 2
    rk = np.sqrt(err / np.sum(sq_dev))
    w = 1 - min(1, rk)
    w_scores = w * scores
    return w, w_scores


def _score_instance(weights: npt.NDArray[np.float64], w_sum: float) -> npt.NDArray[np.float64]:
    """ Weight the total outlier scores by each feature outlier score
    """
    return np.sqrt(np.sum(weights) / w_sum)


def _fit_attr(X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], reg, n_folds: int = 5) -> (npt.NDArray[np.float64], float):
    """
    Parameters
    ----------
    X : numpy.array
        Training data
    y : numpy.array
        Target values
    n_fold : int
        Number of k-fold splits to perform

    Returns
    -------
    Target outlier score vector and errors
    """
    all_scores = np.array([])
    error = 0

    kf = KFold(n_splits=n_folds)

    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]

        model = reg.fit(X_train, y_train)

        # use test set for feature scores
        pred = model.predict(X[test_idx])
        scores = (y[test_idx] - pred) ** 2
        
        error += np.sum(scores)
        all_scores = np.concatenate((all_scores, scores))

    return all_scores, error


def fit(reg, X, n_folds=5, return_weighted=True):
    """

    """
    w_sum = 0
    w_scores_list = []

    n = X.shape[1]

    # iterate feature columns of matrix
    for i in range(n):
        x, y = _partition_mat(X, i)

        # fit regression using selected column as target
        scores, err = _fit_attr(x, y, reg, n_folds)

        # weight the score and error
        w, w_scores = _weight_scores(y, scores, err)
        w_sum += w

        w_scores_list.append(w_scores)

    w_scores_mat = np.column_stack(w_scores_list)

    if not return_weighted:
        return w_scores_list, w_scores_mat

    return np.apply_along_axis(
        func1d=partial(_score_instance, w_sum=w_sum),
        axis=1,
        arr=w_scores_mat
    )


