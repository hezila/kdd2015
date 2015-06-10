#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import xgboost as xgb

from base_classifier import BaseClassifier

class XGBClassifier(BaseClassifier):
    """

    """

    def __init__(self, max_depth=5,
                    learning_rate=0.1,
                    n_estimators=150,
                    silent=True,
                    objective="binary:logistic",
                    nthread= 8,
                    gamma=0.5,
                    min_child_weight=1,
                    max_delta_step=0,
                    subsample=1,
                    colsample_bytree=1,
                    base_score=0.8,
                    seed=0,
                    missing=None
                    ):

        self.n_estimators = n_estimators
        self.min_child_weight=min_child_weight
        self.max_depth=max_depth
        self.learning_rate = learning_rate
        self.nthread=nthread
        self.base_score = base_score
        self.gamma = gamma
        self.nthread = nthread

    def fpreproc(self, dtrain, param):
        label = dtrain.get_label()
        ratio = float(np.sum(label == 0)) / np.sum(label==1)
        param['scale_pos_weight'] = ratio
        return param

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


        # xgb.cv(param, xgmat_train, num_round, nfold=5, metrics={'error'}, seed = 0)
        clf = xgb.XGBClassifier(max_depth=self.max_depth,
                                learning_rate=self.learning_rate,
                                n_estimators=self.n_estimators,
                                silent=True,
                                objective="binary:logistic",
                                nthread=self.nthread,
                                gamma=self.gamma,
                                min_child_weight=self.min_child_weight,
                                max_delta_step=0,
                                subsample=0.8,
                                colsample_bytree=0.6,
                                base_score=self.base_score,
                                seed=0)

        clf = clf.fit(X, y)

        self.model = clf

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
        pass

    def evaluate(self, X, y):
        pass

    def print_coefficients(self, num_rows=18):
        # coefficients = self.model['coefficients']
        # coefficients.print_rows(num_rows=num_rows)

        pass

if __name__ == "__main__":
    X = np.random.randn(10,3)
    y = np.random.randint(0, 2, 10)
    clf = LogisticClassifier()
    clf.fit(X, y)
    yhat = clf.predict(X)
