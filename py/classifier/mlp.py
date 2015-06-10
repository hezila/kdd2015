#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sknn.mlp import Classifier, Layer


from base_classifier import BaseClassifier

class MLPClassifier(BaseClassifier):
    """

    """

    def __init__(self, verbose=0):

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

        nn = Classifier(
                layers=[
                    # Convolution("Rectifier", channels=10, pool_shape=(2,2), kernel_shape=(3, 3)),
                    Layer('Rectifier', units=100),
                    Layer('Softmax')],
                learning_rate=0.02,
                learning_rule='momentum',
                learning_momentum=0.9,
                batch_size=25,
                valid_set=None,
                # valid_set=(X_test, y_test),
                n_stable=10,
                n_iter=10,
                verbose=True)
        nn.fit(X, y)


        self.model = nn

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
