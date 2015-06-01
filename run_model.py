#!/usr/bin/env python
#-*- coding: utf-8 -*-

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '29-05-2015'


import os, sys
import gc
import pandas as pd
import numpy as np
import itertools
import simplejson as json

from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.decomposition import PCA


from dataset import *
from data import *
from util import *
from model.glbtc import BoostedTreesClassifier
from model.gllgc import LogisticClassifier
from model.glknn import KNNClassifier
from model.glnn import NNClassifier
from model.sksgd import StochasticGradientClassifier
from model.skrfc import RFCClassifier


from optparse import OptionParser


def group_data(data, degree=3, hash=hash):
    """
    numpy.array -> numpy.array

    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in itertools.combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return np.array(new_data).T

def cal_auc(y, preds):
    fpr, tpr, thresholds = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc

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
        model.model = None
        gc.collect()
    return mean_auc/n_folds

def creat_clf(name, paras=None):
    if name == 'sgd':
        alpha = paras['alpha']
        epsilon = paras['epsilon']
        l1_ratio = 0
        n_iter = paras['n_iter']

        sgdcf = StochasticGradientClassifier(alpha = alpha,
                                                epsilon = epsilon,
                                                l1_ratio = l1_ratio,
                                                n_iter = n_iter)
        return sgdcf
    elif name == 'lgc':

        l2_penalty = paras['l2_penalty']
        max_iterations = paras['max_iterations']
        step_size = paras['step_size']
        convergence_threshold = paras['convergence_threshold']
        lgcf = LogisticClassifier(l2_penalty = l2_penalty,
                                    max_iterations = max_iterations,
                                    step_size = step_size,
                                    convergence_threshold = convergence_threshold,
                                    verbose = False)
        return lgcf
    # elif name == 'logistic':
    #     eta = paras['eta']
    #     epochs = paras['epochs']
    #     lambda_ = paras['lambda_']
    #     learning = paras['learning']
    #
    #     logistic = LogisticRegression(eta = eta, epochs = epochs, lambda_ = lambda_, learning = learning)
    #
    #     return logistic

    elif name == 'rfc':
        n_estimators = paras['n_estimators']
        max_depth= paras['max_depth']
        min_samples_split = paras['min_samples_split']
        min_samples_leaf = paras['min_samples_leaf']
        class_weight = None
        if 'class_weight' in paras:
            class_weight = paras['class_weight']
        rfc = RFCClassifier(n_estimators = n_estimators,
                            max_depth = max_depth,
                            min_samples_split = min_samples_split,
                            min_samples_leaf = min_samples_leaf,
                            class_weight = class_weight)
        return rfc
    elif name == 'nn':
        nn = NNClassifier()
        return nn
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

def grid_search(name, paras={}, tr_x=None, tr_y=None, n_folds=5):
    log_info("grid search for %s" % name)
    results = []
    para_list = []

    para_keys = paras.keys()
    n = [len(paras[k]) for k in para_keys]

    inputdata = []
    for k in para_keys:
        inputdata.append(paras[k])

    para_array = list(itertools.product(*inputdata))
    best_i = 0
    best_auc = 0
    # print 'Unreachable objects:',
    for i, param in enumerate(para_array):
        param_dict = {}

        for j, v in enumerate(param):
            k = para_keys[j]
            param_dict[k] = v

        # print "test params", param_dict

        para_list.append(param_dict)
        print param_dict
        clf = creat_clf(name, param_dict)
        auc = cv_loop(tr_x, tr_y, clf, n_folds = n_folds, verbose=False)
        print '%f ' % auc
        if auc > best_auc:
            best_i = i
            best_auc = auc
        results.append(auc)

        n = gc.collect()

    best_params = para_list[best_i]

    log_info(" with best auc: %f" % results[best_i])

    return (best_params, results[best_i])

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
        print '%s: %f' % (name, best_w)

        j += 1

    return weights


def ensembel_fit(candidates, X, y, verbose=False):
    for cname, clf in candidates:
        clf.fit(X, y)
        if verbose:
            print 'finish training: %s' % cname

def ensemble(candidates, test_x, test_y=None, weights=None):
    m = len(candidates)
    preds = None

    if weights is None or len(weights) == 0:
        weights = [1.0 / m] * m

    sum_w = 0.0
    for cname, clf in candidates:
        sum_w += weights[cname]
        yhat = clf.predict_proba(test_x)
        if test_y is not None:
            auc = cal_auc(test_y, yhat[:,1])
            print 'AUC (%s): %f' % (cname, auc)

        if preds is None:
            preds = weights[cname] * yhat[:,1]
        else:
            preds += weights[cname] * yhat[:,1]

    preds = preds / sum_w

    if test_y is not None:
        auc = cal_auc(test_y, preds)
        print 'Ensembe AUC: %f' % auc
    return preds


def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)

    parser.add_option("-t", "--train", dest="train",
        help="the train file")
    parser.add_option("-s", "--test", dest="test", help="the test file")
    parser.add_option("-o", "--output", dest="output", help="the output file")
    parser.add_option("-n", "--nfolds", dest="nfolds", help="the folds number")
    parser.add_option("-m", "--model", dest="model", help="the model name")
    parser.add_option("-f", "--feature", dest="feature",
        help="the feature file")
    parser.add_option("-p", "--paras", dest="paras", help="the paras file")

    (options, remainder) = parser.parse_args()



    train_dataset, test_dataset = load_dataset(options.train, options.test,
                                                                with_norm = False,
                                                                verbose = False)

    # print train_dataset.describe().transpose()
    # sys.exit(0)

    n_folds = 5
    if options.nfolds:
        n_folds = int(options.nfolds)


    tr_label = train_dataset.target.values
    # tr_label, _ = pd.factorize(train_dataset.target)

    tr_data = train_dataset.drop('target', axis=1)
    tr_data = tr_data.drop('eid', axis=1)
    tr_data = tr_data.drop('cid', axis=1)

    tt_data = test_dataset.drop('cid', axis=1)
    tt_ids = tt_data.eid.values
    tt_data = tt_data.drop('eid', axis=1)



    skip_log_ftrs = ["#daytime_ratio",
                        "#night_time",
                        "#day_time"
                        "#weekend_ratio",
                        "#weekend_time",
                        "#weekday_time",
                        "#std_lagging",
                        "#ratio_browser",
                        "#0_6h_request",
                        "#6-9h_request",
                        "#8-12h_request",
                        "#12-18h_request",
                        "#17-20h_request",
                        "#19-24h_request",
                        "#access_pert",
                        "#video_pert",
                        "#discussion_pert",
                        "#wiki_pert",
                        "#problem_pert"]

    cols = tr_data.columns

    log_featurs = [col for col in cols if col not in skip_log_ftrs]
    # log_transf_replace(tr_data, log_featurs)
    # log_transf_replace(tt_data, log_featurs)

    log_transf(tr_data, log_featurs)
    log_transf(tt_data, log_featurs)

    square_root_transf(tr_data, cols)
    square_root_transf(tt_data, cols)

    inverse_transf(tr_data, cols)
    inverse_transf(tt_data, cols)

    # all_data = np.vstack((tr_data, tt_data))
    #
    # pca = PCA(n_components=100)
    # pca.fit(all_data)
    # tr_data = pca.transform(tr_data)
    # tt_data = pca.transform(tt_data)

    # all_data = np.vstack((tr_data.ix[:,1:-1], tt_data.ix[:,1:-1]))
    # num_train = np.shape(tt_data)[0]
    #
    # # Transform data
    # print "Transforming data..."
    # dp = group_data(all_data, degree=2)
    # dt = group_data(all_data, degree=3)
    #
    # X = all_data[:num_train]
    # X_2 = dp[:num_train]
    # X_3 = dt[:num_train]
    #
    # X_test = all_data[num_train:]
    # X_test_2 = dp[num_train:]
    # X_test_3 = dt[num_train:]
    #
    # tr_data = np.hstack((X, X_2, X_3))
    # tt_data = np.hstack((X_test, X_test_2, X_test_3))
    # num_features = tr_data.shape[1]
    # print num_features

    task = 'ensemble'
    if task == 'search':
        m = options.model
        if m == 'lgc':
            search_params = {"l2_penalty": [0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0],
                         "max_iterations": [250],
                         "step_size": [0.01],
                         "convergence_threshold": [0.001]}
        elif m == 'gbt':
            search_params = {"min_loss_reduction": [0.8, 1.0, 1.5],
                    "step_size": [0.01],
                    "min_child_weight": [5, 8, 15, 20],
                    "column_subsample": [0.4],
                    "row_subsample": [0.6],
                    "max_depth": [8],
                    "max_iterations": [500]}
        elif m == 'rfc':
            search_params = {
                    "n_estimators" : [150, 300, 350, 400, 450, 500],
                    "max_depth" : [9, 10],
                    "min_samples_split" : [10, 20],
                    "min_samples_leaf" : [8, 10]
            }

        elif m == 'nn':
            clf = creat_clf('nn')
            auc = cv_loop(tr_data, tr_label, clf, n_folds = n_folds, verbose=False)
            print '%s AUC: %f' % ('NeuralNet', auc)
            sys.exit(0)

        best_param, best_auc = grid_search(m, search_params, tr_data, tr_label, n_folds=n_folds)
        print 'Best PARAS for %s: ' % m,
        print best_param

    elif task == 'valide':
        paras_doc = json.load(open(options.paras, 'r'))
        for m in paras_doc.keys():
            print '%s:' % m
            paras = paras_doc[m]

            clf = creat_clf(m, paras)
            auc = cv_loop(tr_data, tr_label, clf, n_folds = n_folds, verbose=False)
            print 'with AUC: \t%f\n' % (auc)

    elif task == 'fitw':
        paras_doc = json.load(open(options.paras, 'r'))
        cds = []

        nn = creat_clf('nn')
        cds.append(('nn', nn))

        gbt = creat_clf('gbt', paras_doc['gbt'])
        cds.append(('gbt', gbt))

        rfc = creat_clf('rfc', paras_doc['rfc'])
        cds.append(('rfc', rfc))

        lgc = creat_clf('lgc', paras_doc['lgc'])
        cds.append(('lgc', lgc))

        print fit_weight(cds, tr_data, tr_label, n_folds = 3)

    elif task == 'ensemble':
        weights = {'lgc': 0.6, 'gbt': 1.8, 'rfc': 0.6}

        paras_doc = json.load(open(options.paras, 'r'))
        cds = []

        # nn = creat_clf('nn')
        # cds.append(('nn', nn))

        gbt = creat_clf('gbt', paras_doc['gbt'])
        cds.append(('gbt', gbt))

        rfc = creat_clf('rfc', paras_doc['rfc'])
        cds.append(('rfc', rfc))

        # lgc = creat_clf('lgc', paras_doc['lgc'])
        # cds.append(('lgc', lgc))

        ensembel_fit(cds, tr_data, tr_label, verbose=True)
        preds = ensemble(cds, tt_data, None, weights)

        eids = []
        results = {}
        for i, p in enumerate(preds):
            id = int(tt_ids[i])
            eids.append(id)
            results[id] = p

        eids.sort()

        output = open(options.output, 'w')
        for k in eids:
            p = results[k]
            print '%d, %.6f' % (k, p)
            output.write('%d,%.6f\n' % (k, p))
        output.close()


if __name__ == '__main__':
    main()
