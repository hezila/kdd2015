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


def load_dataset(train_file, test_file, with_norm=False, verbose=False, skip_cols=['eid', 'cid', 'target']):
    # import data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # # drop ids and get labels
    # labels = train.target.values
    # train = train.drop('eid', axis=1)

    print train.target.value_counts()/train.target.count()

    # train = train.drop('target', axis=1)
    #
    # test = test.drop('eid', axis=1)

    if verbose:
        for ftr in train.columns:
            if ftr in skip_cols:
                continue
            mean = train[ftr].mean()
            stdev = train[ftr].std()
            maxvalue = train[ftr].max()
            print '%s\t: %f (max: %f; std: %f)' % (ftr, mean, maxvalue, stdev)
            # output =  open('output/%s_hist.txt' % ftr, 'w')
            # output.write('%s' % '\n'.join(['%f' % v for v in train[ftr]]))
            # output.close()

    if with_norm:
        col_features = train.columns
        for ftr in col_features:
            if ftr in skip_cols:
                continue
            mean = train[ftr].mean()
            stdev = train[ftr].std()
            train[ftr] = (train[ftr] - mean) / stdev
            test[ftr] = (test[ftr] - mean) / stdev
            # if verbose:
            #     print '%s\t: %f (std: %f)' % (ftr, mean, stdev)

    return (train, test)

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
    cv = StratifiedKFold(labels, n_folds=n_folds)
    for train_index, test_index in cv:
        train_x, train_y  = dataset.values[train_index], labels[train_index]
        test_x, test_y = dataset.values[test_index], labels[test_index]
        yield (train_x, train_y, test_x, test_y)
