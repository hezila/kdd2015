#!/usr/bin/env python
#-*- coding: utf-8 -*-

import graphlab as gl
import numpy as np
import pandas as pd

from base import *

class NNClassifier(BaseClassifier):
    """

    """

    def __init__(self,
                validation_set= None,
                verbose=False):
        self.validation_set = validation_set

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

        X = gl.SFrame(pd.DataFrame(X))
        X['target'] = y
        # net = gl.deeplearning.get_builtin_neuralnet('mnist')
        net = gl.deeplearning.create(X, target='target')
        self.model = gl.neuralnet_classifier.create(
                X, target = 'target', validation_set = self.validation_set,
                network = net,
                max_iterations = 100,
                verbose = self.verbose)
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
        preds = self.model.predict_topk(X, output_type = 'score',
                                   k = self.num_class)

        preds['row_id'] = preds['row_id'].astype(int) + 1
        preds = preds.unstack(['class', 'score'], 'probs').unpack(
                                'probs', '')
        preds = preds.sort('row_id')
        del preds['row_id']
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

    def print_coefficients(self, num_rows=18):
        pass

if __name__ == "__main__":
    X = np.random.randn(10,3)
    y = np.random.randint(0, 2, 10)
    clf = BoostedTreesClassifier()
    clf.fit(X, y)
    yhat = clf.predict(X)

    p = clf.predict_proba(np.random.randn(1, 3))
    print p
    print p[0][1]
