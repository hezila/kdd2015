#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit


def load_dataset(train_file, test_file):
    # import data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # # drop ids and get labels
    labels = train.target.values
    train = train.drop('eid', axis=1)

    print train.target.value_counts()/train.target.count() 

    train = train.drop('target', axis=1)
    #
    # test = test.drop('eid', axis=1)

    return (train, labels, test)

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
