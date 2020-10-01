import numpy as np


class LogisticRegressionWithWeightsClassifier:
    def __init__(self):
        self.theta = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta, x):
        return np.dot(x, theta)

    def probability(self, theta, x):
        return self.sigmoid(self.net_input(theta, x))

    # ===================================================================================
    # Our new cost function:
    # We multiplied the 2 possible outcomes, i.e. the 2 branches by sample_weight vector.
    # ===================================================================================
    def cost_function(self, x, y, sample_weight):
        m = x.shape[0]
        return -1 / m * np.sum((sample_weight * (y * np.log(self.probability(self.theta, x))
                                + (1 - y) * np.log(1 - self.probability(self.theta, x)))))

    # ===================================================================================
    # Our new gradient:
    # We multiplied (h(x)-y) by sample_weight vector
    # As required from the derivative of the new cost function.
    # ===================================================================================
    def gradient(self, x, y, sample_weight):
        m = x.shape[0]
        return 1 / m * np.dot(x.T, sample_weight * (self.probability(self.theta, x) - y))

    def predict(self, x):
        # ===================================================================================
        # The addition of 1's new column has came inside here
        # ===================================================================================
        updated_x = np.c_[np.ones((x.shape[0], 1)), x]

        # ===================================================================================
        # We need to classify the outcome to 2 groups - labeled 1 and -1, so we adjust it.
        # ===================================================================================
        return np.array([1 if item > 0.5 else -1 for item in self.probability(self.theta, updated_x)])

    def fit(self, X, y, sample_weight=None, lr=0.03, max_iters=100, min_cost=0.05):
        # ===================================================================================
        # We no longer receive theta, but make the training of each new round of Adaboost on
        # random starting theta's.
        # ===================================================================================
        self.theta = np.random.randn(X.shape[1] + 1)

        # ===================================================================================
        # In case we are in the case in which no theta's are given.
        # ===================================================================================
        sample_weight = np.ones(y.shape) if sample_weight is None else sample_weight

        # ===================================================================================
        # The addition of 1's new column has came inside here
        # ===================================================================================
        updated_x = np.c_[np.ones((X.shape[0], 1)), X]

        # ===================================================================================
        # The Sigmoid function is working with values (0,1) so we adjust our given labels to
        # use with this function.
        # ===================================================================================
        y = np.array([1 if item == 1 else 0 for item in y])

        # ===================================================================================
        # Our implementation of "fmin_tnc" library function, i.e. implementation of "Gradient
        # descent" algorithm - We try to perform pruning when current achieved cost is
        # getting less than Minimal cost which is supplied to the algorithm or if the cost is
        # starting to increase instead of decrease.
        # ===================================================================================
        iters = 0
        prev_cost = self.cost_function(updated_x, y, sample_weight)
        while iters < max_iters:
            self.theta -= lr * self.gradient(updated_x, y, sample_weight)
            cost = self.cost_function(updated_x, y, sample_weight)
            if cost < min_cost or cost > prev_cost:
                break
            prev_cost = cost
            iters += 1

        return self
