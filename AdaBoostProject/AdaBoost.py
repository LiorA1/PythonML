from plotting import plot_adaboost, plot_staged_adaboost
from UpdatedLogisticRegression import LogisticRegressionWithWeightsClassifier
import numpy as np

# =================================================================================
# The only real change that was made in this file is the change of line 48.
# All the other changes are changes of variable names, to be more appropriate ones.

# Shortcuts and Acronyms:
# =======================
# LogisticRegressionWithWeightsClassifiers ==> LRWC
# =================================================================================


class AdaBoost:
    """ AdaBoost ensemble classifier from scratch """

    def __init__(self):
        self.LRWC = None
        self.LRWC_weights = None
        self.errors = None
        self.sample_weights = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Response variable must be ±1'
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
        """ Fit the model using training data """

        X, y = self._check_X_y(X, y)
        n = X.shape[0]

        # init numpy arrays
        self.sample_weights = np.zeros(shape=(iters, n))
        self.LRWC = np.zeros(shape=iters, dtype=object)
        self.LRWC_weights = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)

        # initialize weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(iters):
            # fit  weak learner
            curr_sample_weights = self.sample_weights[t]
            lrwc = LogisticRegressionWithWeightsClassifier()
            lrwc = lrwc.fit(X, y, sample_weight=curr_sample_weights)

            # calculate error and stump weight from weak learner prediction
            lrwc_pred = lrwc.predict(X)

            err = curr_sample_weights[(lrwc_pred != y)].sum()  # / n
            lrwc_weight = np.log((1 - err) / err) / 2

            # update sample weights
            new_sample_weights = (
                    curr_sample_weights * np.exp(-lrwc_weight * y * lrwc_pred)
            )

            new_sample_weights /= new_sample_weights.sum()

            # If not final iteration, update sample weights for t+1
            if t + 1 < iters:
                self.sample_weights[t + 1] = new_sample_weights

            # save results of iteration
            self.LRWC[t] = lrwc
            self.LRWC_weights[t] = lrwc_weight
            self.errors[t] = err

        return self

    def predict(self, X):
        """ Make predictions using already fitted model """
        lrwc_preds = np.array([lrwc.predict(X) for lrwc in self.LRWC])
        return np.sign(np.dot(self.LRWC_weights, lrwc_preds))

