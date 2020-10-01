import numpy as np


# https://stackoverflow.com/questions/32675024/getting-pycharm-to-import-sklearn
from sklearn.datasets import make_gaussian_quantiles

from AdaBoost import AdaBoost

from plotting import plot_staged_adaboost


def make_toy_dataset(n: int = 100, random_seed: int = None):
    """ Generate a toy dataset for evaluating AdaBoost classifiers """

    if random_seed:
        np.random.seed(random_seed)

    x, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2)

    # Original labels are {0, 1}, after performing ({0, 1}*2) - 1
    # we get {-1, 1} as we need in AdaBoost algorithm.
    return x, y * 2 - 1



iterations = 11
X, y = make_toy_dataset(n=10, random_seed=10)
clf = AdaBoost().fit(X, y, iters=iterations)

# plot_adaboost(X, y, clf)
plot_staged_adaboost(X, y, clf, iters=iterations)

train_err = (clf.predict(X) != y).mean()
print(f'Train error: {train_err:.1%}')