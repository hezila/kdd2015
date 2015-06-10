#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.feature_selection import RFE


from base_classifier import BaseClassifier

class LogisticClassifier(BaseClassifier):
    """

    """

    def __init__(self, penalty='l2',
                    dual = False,
                    tol = 0.0001,
                    C = 1.0, # smaller values specify stronger regularization
                    fit_intercept = True,
                    intercept_scaling = 1,
                    class_weight = "auto",
                    random_state = None,
                    solver = 'liblinear', # ‘newton-cg’, ‘lbfgs’, ‘liblinear’
                    max_iter = 100,
                    multi_class = 'ovr',
                    verbose=0):

        self.penalty = penalty
        self.solver = solver
        self.tol = tol
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.intercept_scaling = intercept_scaling
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
        model = linear_model.LogisticRegression(penalty = self.penalty,
                                                        solver = self.solver,
                                                        tol = self.tol,
                                                        max_iter = self.max_iter,
                                                        intercept_scaling = self.intercept_scaling,
                                                        class_weight = self.class_weight,
                                                        C=self.C)

        # rfe = RFE(model, 50)
        # rfe = rfe.fit(X, y)
        # # summarize the selection of the attributes
        # print(rfe.support_)
        # print(rfe.ranking_)
        # self.model = rfe

        self.model = model.fit(X, y)

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
