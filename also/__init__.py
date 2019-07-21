from functools import partial

import numpy as np

from sklearn.model_selection import KFold

from also import utils


def fit(reg, X, n_folds=5):
    """

    """
    w_sum = 0
    w_scores_list = []

    n = X.shape[1]

    for i in range(n):
        x, y = utils.partition_mat(X, i)

        scores, err = _fit_attr(x, y, reg, n_folds)
        w, w_scores = utils.weight_scores(y, scores, err)
        w_sum += w

        w_scores_list.append(w_scores)

    w_scores_mat = np.column_stack(w_scores_list)

    _score_inst = partial(utils.score_instance, w_sum=w_sum)
    return np.apply_along_axis(
        func1d=_score_inst,
        axis=1,
        arr=w_scores_mat
    )


def _fit_attr(X, y, reg, n_folds=5):
    all_scores = None
    error = 0

    kf = KFold(n_splits=n_folds)

    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]

        model = reg.fit(X_train, y_train)

        for tidx in test_idx:
            pred = model.predict(np.array([X[tidx]]))
            scores = (y[tidx] - pred) ** 2

            error = error + scores.item()

            all_scores = (
                scores
                if all_scores is None
                else np.concatenate((all_scores, scores), axis=0)
            )

    return all_scores, error


def _predicate_instance(model, x, y):
    """
    """
    pred = model.predict(x)
    score = (y - pred) ** 2
    return score


    


