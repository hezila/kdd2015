
#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier

from base import *

class StochasticGradientClassifier(BaseClassifier):
    """
    Wrapper of Graph Lab Logistic Classifier

    Parameters
    ----------
    max_iterations : int, optional
        The maximum number of iterations for boosting. Each iteration results
        in the creation of an extra tree.

    max_depth : float, optional
        Maximum depth of a tree.


    """

    def __init__(self, alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
            eta0=0.01, fit_intercept=True, l1_ratio=0.15,
            learning_rate='optimal', loss='modified_huber', n_iter=5, n_jobs=1,
            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
            verbose=0, warm_start=False):
        self.alpha = alpha
        self.average = average
        self.class_weight = class_weight
        self.epsilon = epsilon
        self.eta0 = eta0
        self.fit_intercept = fit_intercept
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.loss = loss
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.penalty = penalty
        self.power_t = power_t
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.warm_start = warm_start






    def fit(self, X, y):
        """
        Fit the model according to the given training data

        Parameters
        ----------
        X: {array-like}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features

        y: array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            return self.
        """
        self.model = SGDClassifier(alpha=self.alpha, average=self.average, class_weight=self.class_weight,
                                    epsilon=self.epsilon, eta0=self.eta0, fit_intercept=self.fit_intercept,
                                    l1_ratio=self.l1_ratio, learning_rate=self.learning_rate, loss=self.loss,
                                    n_iter=self.n_iter, n_jobs=self.n_jobs, penalty=self.penalty,
                                    power_t=self.power_t, random_state=self.random_state, shuffle=self.shuffle,
                                    verbose=self.verbose, warm_start=self.warm_start)


        self.model.fit(X, y)
        self.num_class = len(np.unique(y))

    def predict_proba(self, X):
        """
        Probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model
        """

        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Class estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        yhat: array-like, shape = (n_samples, )
            Returns the predicted class of the sample
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def print_coefficients(self, num_rows=18):
        print('Intercept: \n', self.model.intercept_)
        print self.model.coef_


if __name__ == "__main__":
    X = np.random.randn(10,3)
    y = np.random.randint(0, 2, 10)
    clf = StochasticGradientClassifier()
    clf.fit(X, y)
    yhat = clf.predict(X)
    print yhat
