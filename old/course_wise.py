#/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import gc
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import auc, roc_curve, roc_auc_score

from dataset import *
from data import *
from util import *
from model.glbtc import BoostedTreesClassifier
from model.gllgc import LogisticClassifier
from model.glknn import KNNClassifier
from model.logistic import LogisticRegression

from model.sksgd import StochasticGradientClassifier

from optparse import OptionParser

import simplejson as json
# List of candidate family classifiers with parameters
# [name, classifier object].
def candidate_families(filename=None, verbose=False):
    candidates = []

    params = None
    if filename:
        para_doc = json.load(open(filename, 'r'))

        sgd_para = para_doc['sgd']
        alpha = sgd_para['alpha']
        epsilon = sgd_para['epsilon']
        l1_ratio = sgd_para['l1_ratio']
        n_iter = sgd_para['n_iter']
        sgdcf = StochasticGradientClassifier(alpha = alpha,
                                                epsilon = epsilon,
                                                l1_ratio = l1_ratio,
                                                n_iter = n_iter)
        candidates.append(["SGD", sgdcf])

        lgc_para = para_doc['lgc']
        l2_penalty = lgc_para['l2_penalty']
        max_iterations = lgc_para['max_iterations']
        step_size = lgc_para['step_size']
        convergence_threshold = lgc_para['convergence_threshold']
        lgcf = LogisticClassifier(l2_penalty = l2_penalty,
                                    max_iterations = max_iterations,
                                    step_size = step_size,
                                    convergence_threshold = convergence_threshold,
                                    verbose = False)
        candidates.append(["LGC", lgcf])


        gbt_para = para_doc['gbt']
        min_loss_reduction = gbt_para['min_loss_reduction']
        step_size = gbt_para['step_size']
        min_child_weight = gbt_para['min_child_weight']
        column_subsample = gbt_para['column_subsample']
        row_subsample = gbt_para['row_subsample']
        max_depth = gbt_para['max_depth']
        max_iterations = gbt_para['max_iterations']
        gbtcf = BoostedTreesClassifier(min_loss_reduction = min_loss_reduction,
                                        class_weights = None,
                                        step_size = step_size,
                                        min_child_weight = min_child_weight,
                                        column_subsample= column_subsample,
                                        row_subsample = row_subsample,
                                        max_depth = max_depth,
                                        max_iterations  = max_iterations)
        candidates.append(["GBT", gbtcf])


    else:

        sgdcf = StochasticGradientClassifier(alpha = 0.5,
                                                epsilon = 0.001,
                                                l1_ratio = 0.0,
                                                n_iter = 250)
        candidates.append(["SGD", sgdcf])

        # knncf = KNNClassifier(verbose = True)
        # candidates.append(["KNN", knncf])


        lgcf = LogisticClassifier(l2_penalty = 0.2,
                                    max_iterations = 250,
                                    step_size = 0.2,
                                    convergence_threshold = 0.001,
                                    verbose = False)
        candidates.append(["LGC", lgcf])



        gbtcf = BoostedTreesClassifier(min_loss_reduction = 0.6,       class_weights = None,
                                        step_size       = .1,    min_child_weight = 10,
                                        column_subsample= 0.6,   row_subsample    = .6,
                                        max_depth       = 5,     max_iterations  = 100)
        candidates.append(["GBT", gbtcf])

    return candidates

def cal_auc(y, preds):
    fpr, tpr, thresholds = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def xval(classifier, dataset, labels, n_folds = 3, verbose=True):
    # log_info('Crossvalidation started... ')

    avg_quality = 0.0
    i = 0
    for tr_x, tr_y, va_x, va_y in folds_split(dataset, labels, n_folds= n_folds):
        i += 1
        classifier.fit(tr_x, tr_y)

        yhat = classifier.predict_proba(va_x)

        preds = yhat[:,1]

        # fpr, tpr, thresholds = roc_curve(va_y, preds)
        # roc_auc = auc(fpr, tpr)
        roc_auc = cal_auc(va_y, preds)

        avg_quality += roc_auc
        del classifier.model
        # log_info('Quality of split... ' + str(quality))
        # print 'AUC in fold %d: %f' % (i, roc_auc)
    quality = avg_quality / n_folds
    # log_info('Estimated quality of model... ' + str(quality))
    if verbose:
        print 'MEAN AUC: %.3f' % quality
    return quality

def ensembel_fit(candidates, X, y):
    for cname, clf in candidates:
        clf.fit(X, y)

def ensembel(candidates, test_x, test_y=None, weights=None):
    m = len(candidates)
    preds = None
    if weights is None or len(weights) == 0:
        weights = [1.0 / m] * m

    for cname, clf in candidates:
        yhat = clf.predict_proba(test_x)
        if test_y is not None:
            auc = cal_auc(test_y, yhat[:,1])
            print 'AUC (%s): %f' % (cname, auc)

        if preds is None:
            preds = weights[cname] * yhat[:,1]
        else:
            preds += weights[cname] * yhat[:,1]

    if test_y is not None:
        auc = cal_auc(test_y, preds)
        print 'Ensembe AUC: %f' % auc
    return preds

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

        clf = creat_clf(name, param_dict)
        auc = xval(clf, tr_x, tr_y, n_folds = n_folds, verbose=False)
        print '%f ' % auc,
        if auc > best_auc:
            best_i = i
            best_auc = auc
        results.append(auc)

        n = gc.collect()
        # print n,
        # print ' ',
    # find the best
    # print 'the best paras: ',
    best_params = para_list[best_i]
    #
    # print para_list[best_i]
    log_info(" with best auc: %f" % results[best_i])

    return (best_params, results[best_i])

def creat_clf(name, paras):
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
    elif name == 'logistic':
        eta = paras['eta']
        epochs = paras['epochs']
        lambda_ = paras['lambda_']
        learning = paras['learning']

        logistic = LogisticRegression(eta = eta, epochs = epochs, lambda_ = lambda_, learning = learning)

        return logistic

    elif name == 'gbt':
        min_loss_reduction = paras['min_loss_reduction']
        step_size = paras['step_size']
        min_child_weight = paras['min_child_weight']
        column_subsample = paras['column_subsample']
        row_subsample = paras['row_subsample']
        max_depth = paras['max_depth']
        max_iterations = paras['max_iterations']
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
    for tr_x, tr_y, va_x, va_y in folds_split(X, y, n_folds= n_folds):
        clf_preds = {}
        for name, clf in cds:
            clf.fit(tr_x, tr_y)
            yhat = clf.predict_proba(va_x)
            clf_preds[name] = yhat[:,1]

        folds_preds.append((va_y, clf_preds))

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
        candi_weights = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.8, 2.0]

        add_preds = []
        for n, va_y, preds in enumerate(folds_preds):
            add_preds.append(preds[name])

        best_w = 0
        best_auc = 0
        for w in candi_weights:
            new_w = sw + w
            quality = 0.0
            for n, va_y, preds in enumerate(folds_preds):
                new_pred = (cul_preds[n] + w * add_preds[n]) / (sw + w)

                roc_auc = cal_auc(va_y, preds)
                quality += roc_auc
            quality = quality / n_folds

            if quality > best_auc:
                best_auc = quality
                best_w = w

        for n, va_y, preds in enumerate(folds_preds):
            cul_preds[n] += w * add_preds[n]
            sw += w
            weights[name] = w

        j += 1

    return weights


def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)

    # parser.add_option("-e", "--train_enroll", dest="train_erlm",
    #     help="the train enrollments file")
    # parser.add_option("-q", "--test_enroll", dest="test_erlm",
    #     help="the test enrollments file")

    parser.add_option("-t", "--train", dest="train",
        help="the train file")
    parser.add_option("-s", "--test", dest="test", help="the test file")
    parser.add_option("-o", "--output", dest="output", help="the output file")
    parser.add_option("-n", "--nfolds", dest="nfolds", help="the folds number")
    parser.add_option("-c", "--course", dest="course", help="the course index")
    # parser.add_option("-i", "--ids", dest="ids", help="the ids file")


    (options, remainder) = parser.parse_args()



    train_dataset, test_dataset = load_dataset(options.train, options.test,
                                                                with_norm = True,
                                                                verbose = False)

    # print train_dataset.describe().transpose()
    #
    # sys.exit(0)

    courses = np.unique(train_dataset.cid.values)

    tr_datas = {}
    tr_labels = {}
    tt_datas = {}
    tt_ids = {}
    for cid in courses:
        tr_data = train_dataset[(train_dataset.cid == cid)]
        # o_data = train_dataset[(train_dataset.cid != cid) & (train_dataset.target == 0)]
        # rows = np.random.choice(o_data.index.values, int(data.shape[0] * 0.1))
        #
        # o_data = o_data.ix[rows]
        #
        # tr_data = pd.concat([data, o_data], ignore_index=True)

        tr_label = tr_data.target.values

        # print tr_data.target.value_counts()/tr_data.target.count()

        tr_data = tr_data.drop('target', axis=1)
        tr_data = tr_data.drop('eid', axis=1)
        tr_data = tr_data.drop('cid', axis=1)
        tr_datas[cid] = tr_data
        tr_labels[cid] = tr_label

        tt_data = test_dataset[test_dataset.cid == cid]
        tt_data = tt_data.drop('cid', axis=1)
        tt_ids[cid] = tt_data.eid.values
        tt_data = tt_data.drop('eid', axis=1)

        tt_datas[cid] = tt_data

    train_dataset = train_dataset.drop('eid', axis=1)
    train_dataset = train_dataset.drop('cid', axis=1)
    train_labels = train_dataset.target.values
    train_dataset = train_dataset.drop('target', axis=1)

    test_dataset = test_dataset.drop('cid', axis=1)
    test_dataset = test_dataset.drop('eid', axis=1)


    nfolds = 5
    if options.nfolds:
        nfolds = int(options.nfolds)

    course = 0
    # cds = candidate_families()
    # task = 'grid_search'
    task = 'pred'
    if task == 'grid_search':
        # init_logging("course_wise.log")
        search_params = {}
        search_params['sgd'] = {"alpha": [1.0, 3.0, 4.0, 6.0, 6.5, 7.0, 7.5, 8.0, 9, 10],
                "epsilon": [0.1, 0.01, 0.05, 0.005, 0.001],
                'n_iter': [100, 300, 500, 800, 1000]
                }

        search_params['lgc'] = {"l2_penalty": [1.0, 5.0, 7.0, 10.0, 12.0, 15.0],
                     "max_iterations": [250, 300, 500],
                     "step_size": [0.01, 0.03, 0.05, 0.1],
                     "convergence_threshold": [0.001]}

        search_params['logistic'] = {
                    "eta": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
                    'epochs': [50, 80, 100, 150, 200, 300, 500, 1000],
                    'learning': ["sgd", "gd"],
                    "lambda_": [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0]
        }

        search_params['gbt'] = {"min_loss_reduction": [0.8],
                "step_size": [0.01],
                "min_child_weight": [5],
                "column_subsample": [0.8],
                "row_subsample": [0.6],
                "max_depth": [5],
                "max_iterations": [200]}

        best_course_params = {}

        i = 0
        for cid in courses:
            # if i != course:
            #     continue
            log_info('validate model for %d-th course: %s' % (i+1, cid))
            i += 1

            tr_x = tr_datas[cid]
            tr_y = tr_labels[cid]

            best_params = {}
            # for name in ['lgc', 'sgd', 'gbt']:
            for name in ['gbt']:
                best_param, best_auc = grid_search(name, search_params[name], tr_x, tr_y, n_folds=5)
                best_params['auc'] = best_auc
                best_params['name'] = name
                best_params['paras'] = best_param
                best_params['cid'] = cid
                print best_param
                print
            # best_course_params[cid] = best_params

                output = open("course_paras/course_%s_%s_params.json" % (cid, name), 'w')
                output.write(json.dumps(best_params))
                output.close()
    elif task == 'states':
        best_params = {}
        for root, dirs, filenames in os.walk("course_paras"):
            for filename in filenames:
                json_doc = json.load(open(os.path.join(root, filename), 'r'))
                cid = json_doc['cid']
                name = json_doc['name']
                if name != 'lgc':
                    continue
                auc = json_doc['auc']
                size = len(tr_labels[cid])
                tr_data = tr_datas[cid]
                pn = tr_data[tr_data.target == 1].target.count() + 0.0
                ratio = pn/tr_data.target.count()

                print '%d,%f,%f' % (size, ratio, auc)

    elif task == 'pred':
        best_params = {}
        for root, dirs, filenames in os.walk("course_paras"):
            for filename in filenames:
                json_doc = json.load(open(os.path.join(root, filename), 'r'))
                cid = json_doc['cid']
                name = json_doc['name']
                if cid not in best_params:
                    best_params[cid] = {}
                paras = json_doc['paras']
                paras['auc'] = json_doc['auc']
                best_params[cid][name] = paras
        clfs = {}
        for cid in best_params.keys():
            params = best_params[cid]
            best_auc = -1
            for name in params.keys():
                ps = params[name]
                auc = ps['auc']
                if auc > best_auc:
                    print name
                    clf = creat_clf(name, ps)
                    clf.fit(tr_datas[cid], tr_labels[cid])

                    clfs[cid] = clf
                    best_auc = auc

        results = {}
        eids = []
        i = 0
        for cid in courses:
            i += 1

            clf = clfs[cid]
            yhat = clf.predict_proba(tt_datas[cid])[:,1]


            for i, p in enumerate(yhat):

                id = int(tt_ids[cid][i])
                eids.append(id)
                results[id] = p

            print yhat[:10]
        eids.sort()
        local_preds = []
        for k in eids:
            p = results[k]
            local_preds.append(p)

        # print local_preds[:10]

        gclfs = candidate_families("gclf_paras.json")
        weights = {"SGD": 0.05, "LGC": 0.1, "GBT": 0.85}

        ensembel_fit(gclfs, train_dataset, train_labels)
        preds = ensembel(gclfs, test_dataset, None, weights)

        output = open(options.output, 'w')
        for i, k in enumerate(eids):
            p = results[k]
            p = 0.85 * p + 0.15 * preds[i]
            # print '%d, %.6f' % (k, p)
            output.write('%d,%.6f\n' % (k, p))
        output.close()


    elif task == 'ensemble':
        i = 0
        for cid in courses():
            print 'ensemble models for %d-th course: %s' % (i+1, cid)
            i += 1
            print '> weights: ',
            weights = {"SGD": 0.05, "LGC": 0.1, "GBT": 0.85}
            print weights
            tr_x, tr_y, va_x, va_y = random_split(tr_datas[cid], tr_labels[cid])
            ensembel_fit(cds, tr_x, tr_y)
            preds = ensemble(cds, va_x, va_y, weights)

    elif task == 'test':
        results = {}
        eids = []
        i = 0
        for cid in courses():
            print 'Test for %d-th course %s' % (i+1, cid)
            i += 1

            print '> weights',
            weights = {"SGD": 0.05, "LGC": 0.1, "GBT": 0.85}
            print weights

            ensembel_fit(cds, tr_datas[cid], tr_labels[cid])
            preds = ensemble(cds, tt_datas[cid], None, weights)

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
