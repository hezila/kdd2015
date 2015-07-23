#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd

from util import *
from dataset import *


train_paths = [
    "../data/train_simple_feature.csv",
    "../data/train_plus_feature.csv",
    "../data/train_azure_plus_feature.csv"
    ]

label_path = "../data/truth_train.csv"

test_paths = [
    "../data/test_simple_feature.csv",
    "../data/test_plus_feature.csv",
    "../data/test_azure_plus_feature.csv"
    ]

train = merge_features(train_paths, label_path)
train = train.drop(['user_drop_ratio'], axis=1)

y = encode_labels(train.dropout.values)
train = train.drop('dropout', axis=1)
tr_ids = train.enrollment_id.values
X = train.drop('enrollment_id', axis=1)
m, n = X.shape
print 'train.shape=%s' % (str(X.shape))

test = merge_features(test_paths)
test = test.drop(['user_drop_ratio'], axis=1)
tt_ids = test.enrollment_id.values
X_test = test.drop('enrollment_id', axis=1)
print 'test.shape=%s' % (str(X_test.shape))
