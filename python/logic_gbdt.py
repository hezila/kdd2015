#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd

from util import *
from dataset import *

def write_submission(tt_ids, preds, file_path):
    eeids = []
    results = {}
    for i, p in enumerate(preds):
        id = tt_ids[i]
        eeids.append(id)
        results[id]=p
    eeids.sort()

    output = open(file_path, 'w')
    for k in eeids:
        p = results[k]
        output.write('%d,%f\n' % (k, p))
    output.close()


train_paths = [
    "../data/train_simple_feature.csv",
    #"../data/train_plus_feature.csv",
    #"../data/train_azure_plus_feature.csv"
    "../data/train_azure_feature.csv",
    "../data/train_module_feature.csv",
    "../data/train_course_feature.csv"
    ]

label_path = "../data/truth_train.csv"

test_paths = [
    "../data/test_simple_feature.csv",
    #"../data/test_plus_feature.csv",
    #"../data/test_azure_plus_feature.csv"
    "../data/test_azure_feature.csv",
    "../data/test_module_feature.csv",
    "../data/train_course_feature.csv"
    ]

train = merge_features(train_paths, label_path)
train = train.drop(['user_drop_ratio'], axis=1)
print 'before.train.shape=%s' % (str(train.shape))


d = 19

day1_v = train.loc[np.logical_and(train['active_days'] <= 2, train['duration'] <= 3)]
day1_y = encode_labels(day1_v.dropout.values)
day1_X = day1_v.drop(['enrollment_id', 'dropout'], axis=1).values

filter_v = train.loc[np.logical_and(train['active_days'] > d, True)]
filter_y = encode_labels(filter_v.dropout.values)
filter_X = filter_v.drop(['enrollment_id', 'dropout'], axis=1).values

train = train.loc[np.logical_and(train['active_days'] <= d, train['active_days'] > 1)]

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


import xgboost as xgb

n_trees = 1000

param = {'max_depth':20, 'eta':.01, 'objective':'binary:logistic',
         'subsample':1.0, 'min_child_weight':100, 'gamma':0.0,
         'nthread': 8, 'colsample_bytree':.5, 'base_score':0.8, 'seed': 999,
         'scale_pos_weight': 3.0 / 7.0, 'silent': 1}
dm_train = xgb.DMatrix(X.values, label=y)
dm_day1 = xgb.DMatrix(day1_X, label=day1_y)
dm_filter = xgb.DMatrix(filter_X, label=filter_y)
dtest = xgb.DMatrix(X_test.values)
test_preds = 0

filter = np.logical_and(X_test.active_days.values == 1, True)


for tr_x, tr_y, va_x, va_y in folds_split(day1_X, day1_y, n_folds=5):
    dtrain = xgb.DMatrix(tr_x, label=tr_y)
    dvalid = xgb.DMatrix(va_x, label=va_y)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    plst = list(param.items()) + [('eval_metric', 'logloss'), ('eval_metric', 'auc')]
    xgb_clf = xgb.train(plst, dtrain, n_trees, watchlist)

    test_preds += xgb_clf.predict(dtest)

test_preds = test_preds / 5.0

filter_v1 = np.logical_and(X_test.active_days.values > 1, True)
test_preds[filter_v1] = 0.0

n_folds = 5
for tr_x, tr_y, va_x, va_y in folds_split(X, y, n_folds=n_folds):

    dtrain = xgb.DMatrix(tr_x, label=tr_y)
    dvalid = xgb.DMatrix(va_x, label=va_y)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid'), (dm_day1, 'one_day')]

    plst = list(param.items()) + [('eval_metric', 'logloss'), ('eval_metric', 'auc')]
    xgb_clf = xgb.train(plst, dtrain, n_trees, watchlist)

    test_preds[filter_v1] += 1.0 / n_folds * xgb_clf.predict(dtest)[filter_v1]

#test_preds = test_preds / 5.0

#xgb_clf = xgb.train(plst, dm_train, n_trees, [(dm_train, 'train')])
#test_preds = 0.3 * test_preds + xgb_clf.predict(dtest) * 0.7

write_submission(tt_ids, test_preds, "logic_gbdt_submission.csv")
