#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from base_classifier import BaseClassifier

class RFCClassifier(BaseClassifier):
    """
    """

    def __init__(self, n_estimators=500, criterion='entropy',
                    max_depth=5, min_samples_split = 8,
                    min_samples_leaf=3, min_weight_fraction_leaf=0.0,
                    max_features='auto', max_leaf_nodes=None,
                    bootstrap=True, oob_score=False, n_jobs=6,
                    random_state=None, verbose=0, warm_start=False,
                    class_weight="auto"):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight




    def fit(self, X, y):
        """
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

        model = RandomForestClassifier(n_estimators = self.n_estimators,
                                            criterion = self.criterion,
                                            max_depth = self.max_depth,
                                            min_samples_split = self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                            max_features=self.max_features,
                                            max_leaf_nodes=self.max_leaf_nodes,
                                            bootstrap=self.bootstrap,
                                            oob_score=self.oob_score,
                                            n_jobs=self.n_jobs,
                                            random_state=self.random_state,
                                            verbose=0,
                                            warm_start=self.warm_start,
                                            class_weight = self.class_weight)
        # self.model = RandomForestClassifier()

        self.model = model.fit(X, y)
        # display the relative importance of each attribute
        # print(model.feature_importances_)

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

    def coefficients(self, num_rows=18):
        return self.model.feature_importances_


if __name__ == "__main__":
    X = np.random.randn(10,3)
    y = np.random.randint(0, 2, 10)
    clf = StochasticGradientClassifier()
    clf.fit(X, y)
    yhat = clf.predict(X)
    print yhat
