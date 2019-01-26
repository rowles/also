import functools

from . import utils

import numpy as np
from sklearn.model_selection import KFold


class ALSO:
    """ Insert slick comment
    """

    def __init__(self, reg, n_folds=5):
        self.reg = reg
        self.n_folds = n_folds

    def fit(self, X):
        w_sum = 0
        w_scores_list = []

        n = X.shape[1]
        for i in range(n):
            x, y = partition_mat(X, i)

            scores, err = self._fit_attr(x, y)
            w, w_scores = _weight_scores(y, scores, err)
            w_sum += w

            w_scores_list.append(w_scores)

        x = np.column_stack(w_scores_list)

        _score_inst = functools.partial(_score_instance, w_sum=w_sum)
        return np.apply_along_axis(_score_inst, 1, w_scores_mat)

    def _fit_attr(self, X, y):
        """ Return vector of scores and error scalar
        """
        all_scores = None
        error = 0

        kf = KFold(n_splits=self.n_folds)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            m = self.reg.fit(X_train, y_train)

            for tidx in test_index:

                pred = m.predict(X[tidx])
                scores = (y[tidx] - pred) ** 2

                error = error + scores.item()

                all_scores = (
                    scores
                    if all_scores is None
                    else np.concatenate((all_scores, scores), axis=0)
                )

        return all_scores, error
