
#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from base_classifier import BaseClassifier

class SVCClassifier(BaseClassifier):
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

    def __init__(self, C=3.0,
                kernel='rbf',
                degree=3,
                gamma=0.2,
                coef0=0.0,
                shrinking=True,
                probability=True,
                tol=0.001,
                cache_size=200,
                class_weight="auto",
                verbose=False,
                max_iter=-1,
                random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = 100
        self.random_state=random_state





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

        model = SVC(C = self.C, kernel = self.kernel,
                        degree = self.degree, gamma = self.gamma,
                        coef0 = self.coef0, shrinking = self.shrinking,
                        probability = self.probability,
                        tol = self.tol, class_weight = self.class_weight,
                        max_iter = self.max_iter, random_state = self.random_state,
                        verbose = self.verbose)

        self.model = model.fit(X, y)
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
