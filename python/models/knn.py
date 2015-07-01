#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier



from base_classifier import BaseClassifier

class KNNClassifier(BaseClassifier):
    """

    """

    def __init__(self, n_neighbors=5,
                 p = 2,
                 weights = 'uniform', # 'uniform', 'distance'
                 scaler=None,
                 verbose=0):

        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        self.scaler = scaler
        self.verbose = verbose



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
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                   p = self.p,
                                   weights = self.weights,
                                   algorithm="kd_tree")

        if self.scaler:
            X = self.scaler.transform(X)

        self.model = clf.fit(X, y)

        # self.print_coefficients()

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

        if self.scaler:
            X = self.scaler.transform(X)
        preds = self.model.predict_proba(X)
        return preds

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
        pass

    def evaluate(self, X, y):
        pass

    def coefficients(self, num_rows=18):
        coefficients = self.model.coef_
        # coefficients.print_rows(num_rows=num_rows)
        return coefficients


if __name__ == "__main__":
    X = np.random.randn(10,3)
    y = np.random.randint(0, 2, 10)
    clf = LogisticClassifier()
    clf.fit(X, y)
    yhat = clf.predict(X)
