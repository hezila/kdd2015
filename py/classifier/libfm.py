#!/usr/bin/env python
#-*- coding: utf-8 -*-

from base_classifier import BaseClassifier
import numpy as np
import pylibfm

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-05-2015'

class LibFMClassifier(BaseClassifier):
    def __init__(self, num_factors = 10, num_iter=50, validation_size=0.01, verbose = False):

        BaseClassifier.__init__(self, verbose = verbose)
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.validation_size = validation_size
        self.verbose = verbose

    def fit(self, X, y):
        self.model = pylibfm.FM(num_factors = self.num_factors,
                                num_iter = self.num_iter,
                                validation_size = self.validation_size,
                                verbose = self.verbose)
        self.model.fit(X, y)
        print self.predict_proba(X)
        self.num_class = len(np.unique(y))


    def predict_proba(self, X):
        proba = np.ones((X.shape[0], 2), dtype=np.float64)
        proba[:, 1] = self.model.predict(X)
        proba[:, 0] -= proba[:, 1]
        return proba
