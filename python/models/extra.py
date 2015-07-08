#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

from base_classifier import BaseClassifier

class ExtTreeClassifier(BaseClassifier):

    def __init__(self, n_estimators=100, criterion='gini',
                 max_depth=40,
                 max_features = 'auto',
                 min_samples_split=10,
                 min_samples_leaf=5,
                 class_weight = "auto"
                 ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight


    def fit(self, X, y):
        clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                   criterion=self.criterion,
                                   max_depth=self.max_depth,
                                   min_samples_split=self.min_samples_split,
                                   min_samples_leaf=self.min_samples_leaf,
                                   min_weight_fraction_leaf=0.0,
                                   max_features= self.max_features,
                                   max_leaf_nodes=None,
                                   bootstrap=True,
                                   oob_score=False,
                                   n_jobs=3, random_state=None,
                                   verbose=0, warm_start=False,
                                   class_weight=self.class_weight)
        self.model = clf.fit(X, y)
        self.num_class = len(np.unique(y))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)
