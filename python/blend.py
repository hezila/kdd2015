#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
from dataset import *

def train_blend(X, y, X_test, clf_list, n_folds, seed=0):
    np.random.seed(seed)

    n_clf = len(clf_list)

    stratified_KFold = list(folds_indexes(y, n_folds))

    m_train, n = X.shape
    
    blend_train = np.zeros((m_train, n_clf))
    
    m_test, n = X_test.shape
    blend_test = np.zeors((m_test, n_clf))
    
    for i, (cname, clf) in enumerate(clf_list):
        print '%d-th classifier: %s' % (i+1, cname)
        blend_test_subfold = np.zeros((m_test, n_folds))
        for j, (tr, cv) in enumerate(stratified_KFold):
            print '%-th fold' % (j+1)
            X_tr, X_cv = X[tr], X[cv]
            y_tr, y_cv = y[tr], y[cv]
            
            clf.fit(X_tr, y_tr)
            blend_train[cv, i] = clf.predict_proba(X_cv)[:,1]
            blend_test_subfold[:, j] = clf.predict_proba(X_test)[:,1]
        blend_test[:,i] = blend_test_subfold.mean(axis=1)
    return blend_train, blend_test
        
    
        
