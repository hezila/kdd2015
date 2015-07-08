#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import metrics

from dataset import *
from util import *

def train_semi(X, y, X_test, clf, n_folds = 5, n_test_folds = 3, n_runs = 3):
    m_train, n = X.shape

    m_test, n = X_test.shape
    tt_y = np.ones(m_test)

    clf = clf.fit(X, y)
    tt_y = clf.predict(X_test)

    print 'TEST y shape=%s' % (str(tt_y.shape))


    train_KFold = list(folds_indexes(y, n_folds))
    test_KFold = list(folds_indexes(tt_y, n_runs))

    blend_X = None
    Ys = []
    for i in xrange(n_runs):
        print 'Iter [%d/%d]' % (i+1, n_runs)
        test_tr, test_cv = test_KFold[i]
        print test_tr
        test_X_tr = X_test[test_tr]
        test_y_tr = tt_y[test_tr]
        Ys.append(test_y_tr)

        if not blend_X:
            blend_X = test_X_tr

        blend_X = np.vstack((blend_X, test_X_tr))
        blend_y = np.concatenate(Ys)


        for k, (tr, cv) in enumerate(train_KFold):
            X_tr, X_cv = X[tr], X[cv]
            y_tr, y_cv = y[tr], y[cv]

            new_X = np.vstack((X_tr, blend_X))
            new_y = np.vstack((y_tr, blend_y))

            clf = clf.fit(new_X, new_y)
            y_cv_pred = clf.predict_proba(X_cv)

            auc = cal_auc(y_cv, y_cv_pred)
            print 'AUC (%d/%d): %f' % (k, n_folds, auc)
