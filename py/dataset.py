#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math

from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    def __init__(self, features = []):
        self.features = features

    def drop_transform(self, dateset, exludes = []):
        d = dateset.copy()
        for f in exludes:
            d = d.drop(f, axis = 1)
        return d

def merge_features(files, label_file=None):
    data_set = None
    for filepath in files:
        if data_set is None:
            data_set = pd.read_csv(filepath)
        else:
            d = pd.read_csv(filepath)
            data_set = pd.merge(data_set, d, on="enrollment_id")
    if label_file is not None:
        labels = pd.read_csv(label_file)
        data_set = pd.merge(data_set, labels, on="enrollment_id")

    data_set = data_set.drop([
                        'fst_day', 'lst_day',
                        'fst_access_month_3',
                        'fst_access_month_4',
                        'fst_access_month_9',
                        'lst_access_month_3',
                        'lst_access_month_4',
                        'lst_access_month_9'], axis=1)
    return data_set

def load_dataset(feature_file, label_file=None, verbose=False):
    # import data
    data_set = pd.read_csv(feature_file)

    if label_file is not None:
        labels = pd.read_csv(label_file)
        data_set = pd.merge(data_set, labels, on="enrollment_id")

    # drop enrollment_id
    # data_set = data_set.drop('enrollment_id', axis=1)
    # data_set = data_set.drop('fst_day', axis=1)
    # data_set = data_set.drop('lst_day', axis=1)

    data_set = data_set.drop(['enrollment_id',
                        'fst_day', 'lst_day',
                        'fst_access_month_3',
                        'fst_access_month_4',
                        'fst_access_month_9',
                        'lst_access_month_3',
                        'lst_access_month_4',
                        'lst_access_month_9'], axis=1)


    return data_set

class NormTransformer:
    def __init__(self, features, exludes=[]):
        self.features = features
        self.exludes = exludes

    def fit_transform(self, sf):
        self.means = {}
        self.stdevs = {}
        for col in self.features:
            if col in self.exludes:
                continue
            mean = sf[col].mean()
            stdev = sf[col].std()
            self.means[col] = mean
            self.stdevs[col] = stdev

            sf[col] = (sf[col] - mean) / stdev

    def transform(self, sf):
        for col in self.features:
            if col in self.exludes:
                continue
            mean = self.means[col]
            stdev = self.stdevs[col]

            sf[col] = (sf[col] - mean) / stdev


def log_transf(sf, features):
    # Create three new columns with logarithmic transform
    for col in features:
        sf['log-' + col] = sf[col].apply(lambda x: math.log(1 + x))

def log_transf_replace(sf, features):
    for col in features:
        # print col
        sf[col] = sf[col].apply(lambda x: math.log(1.0 + x))


def square_root_transf(sf, features):
    for col in features:
        sf['sqr_' + col] = sf[col].apply(lambda x: math.sqrt(x))

def square_root_transf_replace(sf, features):
    for col in features:
        sf[col] = sf[col].apply(lambda x: math.sqrt(x))

def inverse_transf(sf, features):
    for col in features:
        sf['inverse_' + col] = sf[col].apply(lambda x: 1.0 / (x+1.0))

def inverse_transf_replace(sf, features):
    for col in features:
        sf[col] = sf[col].apply(lambda x: 1.0 / (1.0 + x))


def encode_labels(labels):
    # encode labels
    lbl_enc = preprocessing.LabelEncoder()
    return lbl_enc.fit_transform(labels)

def random_split(dataset, labels, test_size = 0.2):
    # labels = encode_labels(train.target.values)

    ### we need a test set that we didn't train on to find the best weights for combining the classifiers
    sss = StratifiedShuffleSplit(labels, test_size=test_size, random_state=31415)
    for train_index, test_index in sss:
        break
    train_x, train_y  = dataset.values[train_index], labels[train_index]
    test_x, test_y = dataset.values[test_index], labels[test_index]

    return (train_x, train_y, test_x, test_y)

def folds_split(dataset, labels, n_folds=3, random_state = 314159):
    cv = StratifiedKFold(labels, n_folds=n_folds, shuffle=True, random_state=random_state)
    data = dataset
    if isinstance(dataset, pd.DataFrame):
        data = dataset.values
    for train_index, test_index in cv:
        train_x, train_y  = data[train_index], labels[train_index]
        test_x, test_y = data[test_index], labels[test_index]
        yield (train_x, train_y, test_x, test_y)
