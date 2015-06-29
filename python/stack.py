#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import metrics

from dataset import *

def train_stack(X, y, X_test, y_test, clf_list, stacker, n_folds = 5, n_runs = 3):
    Xs = []
    Ys = []
    
    for i in xrange(n_runs):
        print 'Iter [%d/%d]' % (i+1, n_runs)
        stratified_KFold = list(folds_indexes(y, n_folds))
        
        for k, (tr, cv) in enumerate(stratified_KFold):
            X_tr, X_cv = X[tr], X[cv]
            y_tr, y_cv = y[tr], y[cv]

            Xk = []
            Yk = []

            for i, (cname, clf) in enumerate(clf_list):
                clf.fit(X_tr, y_tr)
                y_cv_pred = clf.predict(X_cv)
                Xk.append(y_cv_pred)
            Xk = np.column_stack(Xk)
            Yk = np.array(y_cv)
            
            print '[k=%d] X_train.shape=%s' % (k, str(X_tr.shape))
            print '[k=%d] Y_train.shape=%s' % (k, str(y_tr.shape))
            print '[k=%d] X_cv.shape=%s' % (k, str(X_cv.shape))
            print '[k=%d] y_cv.shape=%s' % (k, str(y_cv.shape))
            print '[k=%d] Xk.shape=%s' % (k, str(Xk.shape))
            print '[k=%d] Yk.shape=%s' % (k, str(Yk.shape))

            Xs.append(Xk)
            Ys.append(Yk)
            
        
    blend_X_tr = np.vstack(Xs)
    blend_y_tr = np.concatenate(Ys)
    
    print 'blend_X_train.shape=%s' % (str(blend_X_train.shape))
    print 'blend_Y_train.shape=%s' % (str(blend_y_tr.shape))

    stacker.fit(blend_X_tr, blend_y_tr)

    for i, (cname, clf) in enumerate(clf_list):
        clf.fit(X, y)
    
    blend_X_test = []
    
    for m, clf in enumerate(clf_list):
        Y_test_pred = model.predict(X_test)
        blend_X_test.append(Y_test_pred)
    
    blend_X_test = np.column_stack(blend_X_test)
    blend_Y_test_pred = stacker.predict(blend_X_test)
    
    print 'blend_X_test.shape=%s' % (str(blend_X_test.shape))
    print 'blend_Y_test_pred.shape=%s' % (str(blend_Y_test_pred.shape))
    
    print 'Score: %s' % (metrics.accuracy_score(y_test, blend_Y_test_pred))
    
