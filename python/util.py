#!/usr/bin/env python
#-*- coding: utf-8 -*-

try:
    import simplejson as json
except:
    import json

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dataset import *
from util import *

from models.logistic import LogisticClassifier
from models.rfc import RFCClassifier
from models.xgb import XGBClassifier
#from models.glbtc import BoostedTreesClassifier
from models.extra import ExtTreeClassifier

from optparse import OptionParser
import gc

def cal_auc(y, preds):
    fpr, tpr, threasholds = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def train_and_evaluate(X, y, clf):
    tr_x, tr_y, tt_x, tt_y = random_split(X, y)
    
    clf.fit(tr_x, tr_y)
    preds = clf.predict_proba(tt_x)[:,1]
    auc = cal_auc(tt_y, preds)
    print 'AUC: %f' % auc

def cv_loop(X, y, model, n_folds=5, verbose=False):
    mean_auc = 0.
    i = 0
    for tr_x, tr_y, va_x, va_y in folds_split(X, y, n_folds= n_folds):
        model.fit(tr_x, tr_y)
        preds = model.predict_proba(va_x)[:,1]
        auc = cal_auc(va_y, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, n_folds, auc)
        mean_auc += auc
        i += 1
        # model.model = None                                                                                                                                                         
        gc.collect()
    mean_auc = mean_auc / n_folds
    return mean_auc

def create_clf(name, paras=None):
    if name == 'lgc':
        if paras:
            penalty = paras.get('penalty', "l2")
            max_iter = paras.get('max_iter', 500)
            C = paras.get('C', 1.0)

            lgc = LogisticClassifier(penalty = penalty,
                                        max_iter = max_iter,
                                        C = C,
                                        verbose = 0)

        else:
            lgc = LogisticClassifier()
        return lgc
    
    elif name == 'rfc':
        if paras:
            n_estimators = paras.get('n_estimators', 1000)
            max_depth= paras.get('max_depth', 5)
            max_features = paras.get('max_features', None) # [None, "auto", "sqrt", "log2"]
            min_samples_split = paras.get('min_samples_split', 10)
            min_samples_leaf = paras.get('min_samples_leaf', 5)
            class_weight = paras.get('class_weight', 'auto')
            criterion = paras.get('criterion', "gini")

            rfc = RFCClassifier(n_estimators = n_estimators,
                                max_depth = max_depth,
                                criterion = criterion,
                                min_samples_split = min_samples_split,
                                min_samples_leaf = min_samples_leaf,
                                class_weight = class_weight
                                )
        else:
            rfc = RFCClassifier()
        return rfc

    elif name == "ext":
        if paras:
            n_estimators = paras.get('n_estimators', 1000)
            criterion = paras.get("criterion", "gini") # entropy and gini
            max_depth = paras.get('max_depth', 40)
            clf = ExtTreeClassifier(n_estimators=n_estimators,
                                    criterion=criterion,
                                    max_depth=max_depth)
        else:
            clf = ExtTreeClassifier()
        return clf

    elif name == "xgb":
        if paras:
            max_depth = paras.get("max_depth", 5)
            learning_rate = paras.get("learning_rate", 0.1)
            n_estimators = paras.get("n_estimators", 500)
            min_child_weight = paras.get('min_child_weight', 3)
            xgb = XGBClassifier(max_depth=max_depth,
                                learning_rate = learning_rate,
                                n_estimators = n_estimators,
                                min_child_weight = min_child_weight)
        else:
            xgb = XGBClassifier()
        return xgb

    elif name == 'gbt':
        min_loss_reduction = paras['min_loss_reduction']
        step_size = paras['step_size']
        min_child_weight = paras['min_child_weight']
        column_subsample = paras['column_subsample']
        row_subsample = paras['row_subsample']
        max_depth = paras['max_depth']
        max_iterations = paras['max_iterations']
        class_weights = None
        if 'class_weights' in paras:
            class_weights = paras['class_weights']
        gbtcf = BoostedTreesClassifier(min_loss_reduction = min_loss_reduction,
                                        class_weights = None,
                                        step_size = step_size,
                                        min_child_weight = min_child_weight,
                                        column_subsample= column_subsample,
                                        row_subsample = row_subsample,
                                        max_depth = max_depth,
                                        max_iterations  = max_iterations)


        return gbtcf
