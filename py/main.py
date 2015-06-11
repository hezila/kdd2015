#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import gc

try:
    import simplejson as json
except:
    import json

from dataset import *

from classifier.libfm import LibFMClassifier
from classifier.logistic import LogisticClassifier
from classifier.rfc import RFCClassifier
from classifier.svc import SVCClassifier
from classifier.pua import PUAdapter
#from classifier.mlp import MLPClassifier
from classifier.xgb import XGBClassifier
from classifier.glbtc import BoostedTreesClassifier

from optparse import OptionParser

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-05-2015'


def cluster_encode(X_train, X_test, codebook='kmeans', k=25):
    if codebook == 'kmeans':
        cb = KMeans(k, n_init=1, init='random')
    elif codebook == 'gmm':
        cb = GMM(n_components=k)
    X = np.vstack((X_train, X_test))
    X = StandardScaler().fit_transform(X)
    print('_' * 80)
    print('fitting codebook')
    print
    print cb
    print
    cb.fit(X)
    print 'fin.'
    X_train = cb.transform(X_train)
    X_test = cb.transform(X_test)
    return X_train, X_test


def cal_auc(y, preds):
    fpr, tpr, threasholds = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def test_clf(X, y, model, verbose=False):
    tr_x, tr_y, tt_x, tt_y = random_split(X, y)

    model.fit(tr_x, tr_y)
    preds = model.predict_proba(tt_x)[:,1]
    auc = cal_auc(tt_y, preds)
    print "AUC: %f" % auc




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
    return mean_auc/n_folds

def create_clf(name, paras=None):
    if name == 'lgc':
        penalty = paras['penalty']
        max_iter = paras['max_iter']
        C = paras['C']
        solver = paras['solver']
        intercept_scaling = paras['intercept_scaling']

        lgcf = LogisticClassifier(penalty = penalty,
                                    max_iter = max_iter,
                                    C = C,
                                    intercept_scaling = intercept_scaling,
                                    solver = solver,
                                    verbose = 1)
        return lgcf
    elif name == 'rfc':
        n_estimators = paras['n_estimators']
        max_depth= paras['max_depth']
        min_samples_split = paras['min_samples_split']
        min_samples_leaf = paras['min_samples_leaf']
        class_weight = paras['class_weight']
        random_state = None
        if 'random_state' in paras:
            random_state = paras['random_state']

        rfc = RFCClassifier(n_estimators = n_estimators,
                            max_depth = max_depth,
                            min_samples_split = min_samples_split,
                            min_samples_leaf = min_samples_leaf,
                            class_weight = class_weight,
                            random_state = random_state)
        return rfc

    elif name == 'svc':
        svc = SVCClassifier()
        return svc

    elif name == 'puc':
        pu = PUAdapter()
        return pu
    # elif name == 'mlp':
    #     mlp = MLPClassifier()
    #     return mlp
    elif name == 'xgb':
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

def fit_weight(cds, X, y, n_folds = 5):
    folds_preds = []
    i = 0
    for tr_x, tr_y, va_x, va_y in folds_split(X, y, n_folds= n_folds):
        clf_preds = {}
        for name, clf in cds:
            clf.fit(tr_x, tr_y)
            yhat = clf.predict_proba(va_x)
            clf_preds[name] = yhat[:,1]

            clf.model = None
            n = gc.collect()

        print '%d/%d folds' % ((i+1), n_folds)
        folds_preds.append((va_y, clf_preds))
        i += 1
    print 'end of fitting ...'

    weights = {}
    name, clf = cds[0]
    weights[name] = 1.0

    sw = 1.0
    cul_preds = []
    for va_y, preds in folds_preds:
        cul_preds.append(preds[name])


    j = 1
    while j < len(cds):
        name, clf = cds[j]
        candi_weights = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.8, 2.0]

        add_preds = []
        for n, fpreds in enumerate(folds_preds):
            va_y, preds = fpreds
            add_preds.append(preds[name])

        best_w = 0
        best_auc = 0
        for w in candi_weights:
            new_w = sw + w
            quality = 0.0
            for n, fpreds in enumerate(folds_preds):
                va_y, preds = fpreds
                new_pred = (cul_preds[n] + w * add_preds[n]) / (sw + w)
                roc_auc = cal_auc(va_y, new_pred)
                quality += roc_auc
            quality = quality / n_folds

            if quality > best_auc:
                best_auc = quality
                best_w = w

        for n, fpreds in enumerate(folds_preds):
            va_y, preds = fpreds
            cul_preds[n] += best_w * add_preds[n]
        sw += best_w
        weights[name] = best_w
        print '%s: %f (%f)' % (name, best_w, best_auc)

        j += 1

    return weights


def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)

    parser.add_option("-t", "--train", dest="train", help="the train file")
    parser.add_option("-l", "--label", dest="label", help="the label file")
    parser.add_option("-s", "--test", dest="test", help="the test file")
    parser.add_option("-n", "--nfolds", dest="nfolds", help="the numbe of folds")

    parser.add_option("-m", "--model", dest="model", help="the model name", default="libfm")
    parser.add_option("-p", "--paras", dest="paras", help="the param json file")

    parser.add_option("-o", "--output", dest="output", help="the output file")

    (options, remainder) = parser.parse_args()

    # train = load_dataset(options.train, options.label)
    train = merge_features(['../data/train_enrollment_feature.csv',
                            '../data/train_course_feature.csv',
                            '../data/train_module_feature.csv'
                            ], options.label)
    train = train.drop('enrollment_id', axis=1)
    y = train.dropout.values
    train_x = train.drop('dropout', axis=1)

    test = merge_features( ['../data/test_enrollment_feature.csv',
                            '../data/test_course_feature.csv',
                            '../data/test_module_feature.csv'])

    tt_ids = test.enrollment_id.values
    test_x = test.drop("enrollment_id", axis=1)

    skip_log_ftrs = ["event_problem_percentage",
                    "event_video_percentage",
                    "event_access_percentage",
                    "event_wiki_percentage",
                    "event_discussion_percentage",
                    "event_nagivate_percentage",
                    "event_page_close_percentage",
                    "fst_access_month_1",
                    "fst_access_month_2",
                    "fst_access_month_3",
                    "fst_access_month_4",
                    "fst_access_month_5",
                    "fst_access_month_7",
                    "fst_access_month_8",
                    "fst_access_month_9",
                    "fst_access_month_10",
                    "fst_access_month_11",
                    "fst_access_month_12",
                    "lst_access_month_1",
                    "lst_access_month_2",
                    "lst_access_month_3",
                    "lst_access_month_4",
                    "lst_access_month_5",
                    "lst_access_month_6",
                    "lst_access_month_7",
                    "lst_access_month_8",
                    "lst_access_month_9",
                    "lst_access_month_10",
                    "lst_access_month_11",
                    "lst_access_month_12",

                    "module_fstaclag_mean",
                    "module_fstaclag_median",
                    "module_fstaclag_min",
                    "module_fstaclag_max",
                    "module_fstaclag_25p",
                    "module_fstaclag_75p",

                    "module_lstaclag_mean",
                    "module_lstaclag_median",
                    "module_lstaclag_min",
                    "module_lstaclag_max",
                    "module_lstaclag_25p",
                    "module_lstaclag_75p",
                    "module_tau",
                    "course_dropratio"
                    ]

    cols = train_x.columns
    log_ftrs = [ftr for ftr in cols if ftr not in skip_log_ftrs]

    #scaler = StandardScaler().fit(np.vstack((X_new, X_test)))
    # scaler = MinMaxScaler(feature_range=(-10,10)).fit(np.vstack((X_new, X_test)))
    # scaler = StandardScaler()
    drops = [
                'fst_access_month_2',
                'fst_access_month_8',
                'fst_access_month_10',
                'lst_access_month_2',
                'lst_access_month_8',
                'lst_access_month_10',
                ]

    # # Select features
    # selector = ExtraTreesClassifier(compute_importances=True, random_state=0)
    # train = selector.fit_transform(train, target)
    # test = selector.transform(test)

    model = options.model
    if model == 'libfm': # to fix it
        num_factors = 10
        num_iter = 15
        validation_size = 0.2
        fm = LibFMClassifier(num_factors=num_factors,
                             num_iter = num_iter,
                             validation_size = validation_size,
                             verbose=True)
        # y = y.astype('float64')
        log_transf_replace(train_x, log_ftrs)

        train_x = normalize(sparse.csr_matrix(train_x))
        # train_x = sparse.csr_matrix(train_x)
        # sss = StratifiedShuffleSplit(y, test_size=0.2, random_state=31415)
        # for train_index, test_index in sss:
        #     break
        # tr_x, tr_y  = train_x[train_index], y[train_index]
        # tt_x, tt_y = train_x[test_index], y[test_index]
        #
        # print y


        fm.fit(train_x, y)

        preds = fm.predict_proba(tt_x)[:,1]
        print preds
        auc = cal_auc(tt_y, preds)
        print 'AUC: %f' % auc

    elif model == 'lgc' or model == 'svc': # logistic regression

        log_transf_replace(train_x, log_ftrs)

        # transf = NormTransformer(cols)
        # transf.fit_transform(train_x)

        train_x = train_x.drop(drops, axis=1)
        test_x = test_x.drop(drops, axis=1)
        cols = train_x.columns

        scaler = StandardScaler(copy=True)  # always copy input data (don't modify in-place)
        # train_x = scaler.fit(train_x).transform(train_x)

        X = np.vstack((train_x, test_x))
        scaler.fit(X)

        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        # Project data through a forest of totall randomized trees
        # and use the leafs the samples end into as a hight-dimensional representation
        # hasher = RandomTreesEmbedding(n_estimators=100)
        # hasher.fit(X)
        # train_x = hasher.transform(train_x)
        # test_x = hasher.transform(test_x)

        # svd = TruncatedSVD(n_components=100)
        # train_x = svd.fit_transform(train_x)

        # tsne = TSNE(n_components=50, random_state=0)
        # train_x = tsne.fit_transform(train_x)

        # new_train_x, new_test_x = cluster_encode(train_x, test_x, codebook="gmm", k=50)
        # train_x = np.hstack((train_x, new_train_x))


        paras = json.load(open(options.paras, 'r'))

        lgc = create_clf(model, paras)

        # test_clf(train_x, y, lgc)

        # print "Features sorted by their coefficients:"
        # print sorted(zip(map(lambda x: round(x, 4), lgc.coefficients()),
        #          cols), reverse=True)

        auc = cv_loop(train_x, y, lgc, n_folds = 5, verbose=False)
        print 'Avg AUC: %f' % auc

    elif model == 'rfc': # random forest classifer
        log_transf_replace(train_x, log_ftrs)

        # transf = NormTransformer(cols)
        # transf.fit_transform(train_x)

        train_x = train_x.drop(drops, axis=1)
        test_x = test_x.drop(drops, axis=1)
        cols = train_x.columns

        # scaler = StandardScaler(copy=True)  # always copy input data (don't modify in-place)
        # # train_x = scaler.fit(train_x).transform(train_x)
        #
        # X = np.vstack((train_x, test_x))
        # scaler.fit(X)
        #
        # train_x = scaler.transform(train_x)
        # test_x = scaler.transform(test_x)

        paras = json.load(open(options.paras, 'r'))

        rfc = create_clf(model, paras)

        # test_clf(train_x, y, rfc)
        #
        # print "Features sorted by their coefficients:"
        # print sorted(zip(map(lambda x: round(x, 4), rfc.coefficients()),
        #          cols), reverse=True)

        auc = cv_loop(train_x, y, rfc, n_folds = 5, verbose=False)
        print 'Avg AUC: %f' % auc

    elif model == 'puc': # PU learning
        pu = create_clf(model, None)

        # train_x = train_x.append(test, ignore_index=True)
        #
        # y = np.append(y, np.zeros(test.shape[0]))
        # y[np.where(y == 1)[0]] = -1
        # y[np.where(y == 0)[0]] = 1
        # y[np.where(y == -1)[0]] = 0

        log_transf_replace(train_x, log_ftrs)

        transf = NormTransformer(cols)
        transf.fit_transform(train_x)

        auc = cv_loop(train_x, y, pu, n_folds = 5, verbose=False)
        print 'Avg AUC: %f' % auc

    elif model == 'xgb': # xgboost
        log_transf_replace(train_x, log_ftrs)

        # transf = NormTransformer(cols)
        # transf.fit_transform(train_x)

        train_x = train_x.drop(drops, axis=1)
        test_x = test_x.drop(drops, axis=1)
        cols = train_x.columns

        scaler = StandardScaler(copy=True)  # always copy input data (don't modify in-place)
        # train_x = scaler.fit(train_x).transform(train_x)

        X = np.vstack((train_x, test_x))
        scaler.fit(X)

        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        # Project data through a forest of totall randomized trees
        # and use the leafs the samples end into as a hight-dimensional representation
        # hasher = RandomTreesEmbedding(n_estimators=100)
        # hasher.fit(X)
        # train_x = hasher.transform(train_x)
        # test_x = hasher.transform(test_x)

        # svd = TruncatedSVD(n_components=100)
        # train_x = svd.fit_transform(train_x)

        # tsne = TSNE(n_components=50, random_state=0)
        # train_x = tsne.fit_transform(train_x)

        # new_train_x, new_test_x = cluster_encode(train_x, test_x, codebook="kmeans", k=50)
        # train_x = np.hstack((train_x, new_train_x))


        xgb = create_clf(model, None)

        # log_transf_replace(train_x, log_ftrs)
        #
        # transf = NormTransformer(cols)
        # transf.fit_transform(train_x)

        auc = cv_loop(train_x, y, xgb, n_folds = 5, verbose=False)
        print 'Avg AUC: %f' % auc
    elif model == 'gbt':
        log_transf_replace(train_x, log_ftrs)

        transf = NormTransformer(cols)
        transf.fit_transform(train_x)

        paras = json.load(open(options.paras, 'r'))

        gbt = create_clf(model, paras)

        auc = cv_loop(train_x, y, gbt, n_folds = 5, verbose=False)
        print 'Avg AUC: %f' % auc

    elif model == 'stacking':
        log_transf_replace(train_x, log_ftrs)
        log_transf_replace(test_x, log_ftrs)
        # transf = NormTransformer(cols)
        # transf.fit_transform(train_x)

        train_x = train_x.drop(drops, axis=1)
        test_x = test_x.drop(drops, axis=1)
        cols = train_x.columns

        scaler = StandardScaler(copy=True)  # always copy input data (don't modify in-place)
        # train_x = scaler.fit(train_x).transform(train_x)

        X = np.vstack((train_x, test_x))
        scaler.fit(X)

        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        tr_x, tr_y, tt_x, tt_y = random_split(train_x, y)


        paras = json.load(open("paras/lgc.json", 'r'))
        lgc = create_clf('lgc', paras)
        lgc.fit(tr_x, tr_y)
        tr_lgc_preds = lgc.predict_proba(tr_x)[:,1]
        tt_lgc_preds = lgc.predict_proba(tt_x)[:,1]

        # train_x['lgc_pred'] = lgc_preds

        paras = json.load(open("paras/rfc.json", 'r'))
        paras['random_state'] = 314
        rfc = create_clf("rfc", paras)
        rfc.fit(tr_x, tr_y)
        tr_rfc_preds1 = rfc.predict_proba(tr_x)[:,1]
        tt_rfc_preds1 = rfc.predict_proba(tt_x)[:,1]
        # train_x['rfc_pred1'] = rfc_preds

        # paras['random_state'] = 159
        # rfc = create_clf("rfc", paras)
        # rfc.fit(train_x, y)
        # preds = rfc.predict_proba(train_x)[:,1]
        # train_x['rfc_pred2'] = preds

        tr_x = np.hstack((tr_x, tr_lgc_preds))
        tr_x = np.hstack((tr_x, tr_rfc_pred2))

        tt_x = np.hstack((tt_x, tt_lgc_preds))
        tt_x = np.hstack((tt_x, tt_rfc_pred2))


        # tr_x['lgc_pred'] = lgc_preds
        # tr_x['rfc_pred1'] = rfc_preds1

        xgb = create_clf('xgb', paras)
        xgb.fit(tr_x, tr_y)
        preds = xgb.predict_proba(tt_x)[:,1]
        auc = cal_auc(tt_y, preds)
        print "AUC: %f" % auc
        # auc = cv_loop(tr_x, tr_y, xgb, n_folds = 5, verbose=False)
        # print 'Avg AUC: %f' % auc

    elif model == 'submit':

        log_transf_replace(train_x, log_ftrs)
        log_transf_replace(test_x, log_ftrs)

        # transf = NormTransformer(cols)
        # transf.fit_transform(train_x)
        #
        # transf.transform(test_x)

        # logistic
        paras = json.load(open("paras/lgc.json", 'r'))
        lgc = create_clf('lgc', paras)
        lgc.fit(train_x, y)

        tr_lgc_preds = lgc.predict_proba(train_x)[:,1]
        # train_x['lgc_pred'] = tr_lgc_preds

        tt_lgc_preds = lgc.predict_proba(test_x)[:,1]
        # test_x['lgc_pred'] = tt_lgc_preds

        paras = json.load(open("paras/rfc.json", 'r'))
        paras['random_state'] = 314
        rfc = create_clf("rfc", paras)
        rfc.fit(train_x, y)

        tr_rfc_preds1 = rfc.predict_proba(train_x)[:,1]
        # train_x['rfc_pred1'] = tr_rfc_preds1

        tt_rfc_preds1 = rfc.predict_proba(test_x)[:,1]
        # test_x['rfc_pred1'] = tt_rfc_preds1

        paras['random_state'] = 159
        rfc = create_clf("rfc", paras)
        rfc.fit(train_x, y)

        tr_rfc_preds2 = rfc.predict_proba(train_x)[:,1]
        # train_x['rfc_pred2'] = tr_rfc_preds2

        tt_rfc_preds2 = rfc.predict_proba(test_x)[:,1]
        # test_x['rfc_pred2'] = tt_rfc_preds2

        paras['random_state'] = 100
        rfc = create_clf("rfc", paras)
        rfc.fit(train_x, y)

        tr_rfc_preds3 = rfc.predict_proba(train_x)[:,1]
        tt_rfc_preds3 = rfc.predict_proba(test_x)[:,1]



        train_x['lgc_pred'] = tr_lgc_preds
        test_x['lgc_pred'] = tt_lgc_preds
        train_x['rfc_pred1'] = tr_rfc_preds1
        test_x['rfc_pred1'] = tt_rfc_preds1
        train_x['rfc_pred2'] = tr_rfc_preds2
        test_x['rfc_pred2'] = tt_rfc_preds2
        train_x['rfc_pred3'] = tr_rfc_preds3
        test_x['rfc_pred3'] = tt_rfc_preds3

        xgb = create_clf('xgb', paras)
        xgb.fit(train_x, y)
        preds = xgb.predict_proba(test_x)[:,1]

        eids = []
        results = {}
        for i, p in enumerate(preds):
            id = int(tt_ids[i])
            eids.append(id)
            results[id] = p

        eids.sort()

        output = open("submission.csv", 'w')
        for k in eids:
            p = results[k]
            # print '%d, %.6f' % (k, p)
            output.write('%d,%.6f\n' % (k, p))
        output.close()

    elif model == 'fitw':
        log_transf_replace(train_x, log_ftrs)
        log_transf_replace(test_x, log_ftrs)
        # transf = NormTransformer(cols)
        # transf.fit_transform(train_x)

        train_x = train_x.drop(drops, axis=1)
        test_x = test_x.drop(drops, axis=1)
        cols = train_x.columns

        scaler = StandardScaler(copy=True)  # always copy input data (don't modify in-place)
        # train_x = scaler.fit(train_x).transform(train_x)

        X = np.vstack((train_x, test_x))
        scaler.fit(X)

        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        cds = []

        xgb = create_clf('xgb', None)
        cds.append(('xgb', xgb))

        paras = json.load(open("paras/rfc.json", 'r'))
        rfc = create_clf("rfc", paras)
        cds.append(('rfc', rfc))

        paras = json.load(open("paras/lgc.json", 'r'))
        lgc = create_clf('lgc', paras)
        cds.append(('lgc', lgc))


        print fit_weight(cds, train_x, y, n_folds = 3)


if __name__ == '__main__':
    main()
