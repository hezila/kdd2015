#!/usr/bin/env python
#-*- coding: utf-8 -*-

import graphlab as gl
import numpy as np
import pandas as pd

from base import *

class LogisticClassifier(BaseClassifier):
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

    def __init__(self):
        pass

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

        X = gl.SFrame(pd.DataFrame(X))
        X['target'] = y
        self.model = gl.logistic_classifier.create(
                X, target='target')


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

        X = gl.SFrame(pd.DataFrame(X))
        preds = self.model.predict_topk(X, output_type = 'probability',
                                   k = self.num_class)
        preds['id'] = preds['id'].astype(int) + 1
        preds = preds.unstack(['class', 'probability'], 'probs').unpack(
                                'probs', '')

        preds = preds.sort('id')
        # print preds['id']

        del preds['id']
        return preds.to_dataframe().as_matrix()

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
        X = gl.SFrame(pd.DataFrame(X))
        yhat = self.model.predict(X)
        return np.array(yhat)

    def evaluate(self, X, y):
        X = gl.SFrame(pd.DataFrame(X))
        X['target'] = y
        return self.model.evaluate(X, metric='accuracy')

if __name__ == "__main__":
    X = np.random.randn(10,3)
    y = np.random.randint(0, 2, 10)
    clf = LogisticClassifier()
    clf.fit(X, y)
    yhat = clf.predict(X)
