#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.feature_selection import RFE


from base_classifier import BaseClassifier
from xgb import XGBClassifier

class WeekClassifier(BaseClassifier):
    def __init__(self, rules={}):
        self.rules = rules
        self.models = {}
        self.dates = {}

    def parse_dates(self, X, y):
        for w in range(2, 4):
            tr_x = X[X['week'] == w]
            tr_y = y[X['week'] == w]
            self.dates[w] = (tr_x, tr_y)

    def fit(self, X, y):
        self.parse_dates(X, y)
        for w in range(2, 4):
            clf = XGBClassifier()
            tr_x, tr_y = self.dates[w]
            clf.fit(tr_x, tr_y)
            self.models[w] = clf

    def predict_proba(self, X, w):
        clf = self.models[w]
        proba = self.clf.predict_proba(X)
        return proba
