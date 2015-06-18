#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys, random
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

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
# from classifier.mlp import MLPClassifier
from classifier.xgb import XGBClassifier
# from classifier.glbtc import BoostedTreesClassifier

from tools.sos import *

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

    train_clusters = cb.predict(X_train)
    test_clusters = cb.predict(X_test)
    # return (train_clusters, test_clusters)
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
        if paras:
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
        else:
            lgcf = LogisticClassifier()
        return lgcf
    elif name == 'rfc':
        if paras:
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
        else:
            rfc = RFCClassifier()
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

    # old_labels = {}
    # with open("../data/week_train_truth.csv.bk", 'r') as r:
    #     for line in r:
    #         eid, l = line.strip().split(',')
    #         if str.isdigit(l):
    #             old_labels[eid] = int(l)
    # print 'old: %d' % len(old_labels)
    # new_labels = {}
    # with open("../data/week_train_truth.csv.bk1", 'r') as r:
    #     for line in r:
    #         eid, l = line.strip().split(',')
    #         if str.isdigit(l):
    #             new_labels[eid] = int(l)
    # cnt = 0.0
    # pos_neg = 0.0
    # neg_pos = 0.0
    #
    # for eid in new_labels.keys():
    #     if eid in old_labels:
    #         ol = old_labels[eid]
    #         nl = new_labels[eid]
    #         # print "%d <> %d" % (ol, nl)
    #
    #         if ol == nl:
    #             cnt += 1
    #         else:
    #             if nl == 1:
    #                 pos_neg += 1.0
    #                 print eid
    #             else:
    #                 neg_pos += 1.0
    #
    # print 'POS>NEG: %d; NEG>POS: %d' % (pos_neg, neg_pos)
    # print 'Correlated: %d/%d (%f)' % (cnt, len(new_labels), cnt / len(new_labels))

    # sys.exit(0)

    # train = load_dataset(options.train, options.label)
    train = merge_features(['../data/train_simple_feature.csv',
                            '../data/train_course_feature.csv',
                            '../data/train_module_feature.csv'
                            ], '../data/truth_train.csv')
    # train = train.drop('enrollment_id', axis=1)
    # y = train.dropout.values
    # train_x = train.drop('dropout', axis=1)

    test = merge_features( ['../data/simple_test.csv',
                            '../data/test_course_feature.csv',
                            '../data/test_module_feature.csv'
                            ])

    tt_ids = test.enrollment_id.values
    # test_x = test.drop("enrollment_id", axis=1)

    skip_log_ftrs = ['enrollment_id',
                    "event_problem_percentage",
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

    cols = train.columns
    log_ftrs = [ftr for ftr in cols if ftr not in skip_log_ftrs]

    #scaler = StandardScaler().fit(np.vstack((X_new, X_test)))
    # scaler = MinMaxScaler(feature_range=(-10,10)).fit(np.vstack((X_new, X_test)))
    # scaler = StandardScaler()




    # # Select features
    # selector = ExtraTreesClassifier(compute_importances=True, random_state=0)
    # train = selector.fit_transform(train, target)
    # test = selector.transform(test)

    models = {}

    # train['week'] = train['enrollment_id'].apply(lambda x: int(x.split('-')[-1][:-1]))
    # test['week'] = test['enrollment_id'].apply(lambda x: int(x.split('-')[-1][:-1]))
    # train['week'] = train['#duration'].apply(lambda x: int(x/7) + 1)
    # test['week'] = test['#duration'].apply(lambda x: int(x/7) + 1)


    dates = {}

    tr = train.drop(['enrollment_id'], axis=1)

    # tr['dropout'] = tr['dropout'].apply(lambda x: int(round(random.random() > 0.4)) if x == 1 else 0)
    tr_y = train.dropout.values

    sample_weight = np.array([4 if i == 0 else 1 for i in tr_y])


    # tr_x = train.drop('dropout', axis=1)
    tr_x = tr.drop(['dropout'], axis=1)

    pos_sum = np.sum(tr_y == 1)
    neg_sum = np.sum(tr_y == 0)

    m, n = tr_x.shape
    print  'M: %d, N: %d, POS: %d; NEG: %d (%f)' % (m, n, pos_sum, neg_sum, neg_sum / (pos_sum + neg_sum + 0.0))

    todrops = [
                # 'year_2014',
                # 'year_2013',
                # 'video_per_day_lst2week',
                # 'video_over10minutes_count',
                # 'session_per_day',
                # 'server_wiki_count_lst2week',
                # 'server_wiki_count_lst1week',
                # 'server_wiki_count',
                # 'server_problem_count_lst1week',
                # 'server_nagivate_count_lst1week',
                # 'server_discussion_count_lst2week',
                # 'server_discussion_count_lst1week',
                # 'server_discussion_count',
                # 'request_weekend_percentage',
                # 'request_weekend_count',
                # 'request_hour_mean',
                # 'problem_over3minutes_count',
                # 'night_time',
                # 'more_weekend',
                # 'module_freq_median',
                # 'lag_nextmodule',
                # 'in_holiday',
                # 'holiday_fstday',
                # 'daytime_ratio',
                # 'day_time',
                # 'browser_video_count_lst2week',
                # 'browser_video_count_lst1week',
                # 'browser_access_count_lst2week',
                # 'browser_access_count_lst1week',
                # 'browser_access_count'
            ]


    tr_x = tr_x.drop(todrops, axis=1)
    cols = tr_x.columns

    # sqrtexp_transf(tr_x, tr_x.columns)

    scaler = MinMaxScaler(copy=True)
    tr_x = scaler.fit_transform(tr_x)

    # O = sos(tr_x, 'euclidean', 30, None)
    # print O

    # svd = TruncatedSVD(n_components=10)
    # tr_svd = svd.fit_transform(tr_x)
    #
    # # tsne = TSNE(n_components=2, random_state=0)
    # # tr_tsne = tsne.fit_transform(tr_svd)
    # tr_x = np.hstack((tr_x, tr_svd))

    # log_transf(tr_x, tr_x.columns)

    # tt_ids = test.enrollment_id.values
    # tt_x = test.drop(['enrollment_id', 'week'], axis=1)

    # clf = create_clf('rfc', None)
    #
    # clf.fit(tr_x, tr_y)
    # print "Features sorted by their coefficients:"
    # print sorted(zip(map(lambda x: round(x, 4), clf.coefficients()),
    #          cols), reverse=True)

    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    # scaler = StandardScaler(copy=True)
    # tr_x = scaler.fit_transform(tr_x)
    # clf = SVC(C=1.0, class_weight='auto', max_iter=200, probability=True, verbose=True)


    clf = create_clf('xgb', None)


    auc = cv_loop(tr_x, tr_y, clf, 5)
    print 'AUC (week=universe): %f' % auc

    sys.exit(0)

    train['week'] = train['duration'].apply(lambda x: int(x/7.0) + 1)
    # test['week'] = test['#duration'].apply(lambda x: int(x/7) + 1)

    for w in range(1, 6):
        print 'week: %d' % w
        eids = train[(train['week'] >= w - 2) & (train['week'] <= w + 2)]['enrollment_id']
        tr = train[train['enrollment_id'].isin(eids)]
        tr = tr.drop(['enrollment_id', 'week'], axis=1)



        tr_y = tr.dropout.values
        tr_y = LabelEncoder().fit_transform(tr_y)

        pos_sum = np.sum(tr_y == 1)
        neg_sum = np.sum(tr_y == 0)
        tr_x = tr.drop('dropout', axis=1)

        # scaler = MinMaxScaler(copy=True)
        # tr_x = scaler.fit_transform(tr_x)

        dates[w] = (tr_x, tr_y)
        m, n = tr_x.shape
        print  'M: %d, N: %d, POS: %d; NEG: %d (%f)' % (m, n, pos_sum, neg_sum, neg_sum / (pos_sum + neg_sum + 0.0))

        clf = create_clf('xgb', None)
        # clf = SVC(C=5.0, class_weight='auto', max_iter=1500, kernel='poly', probability=True, verbose=True)
        # clf = GaussianNB()
        models[w] = clf
        auc = cv_loop(tr_x, tr_y, models[w], 5)
        print 'AUC (w=%d): %f' % (w, auc)

    sys.exit(0)

    preds = clf.predict_proba(tt_x)[:,1]

    eeids = []
    results = {}

    for i, p in enumerate(preds):
        id = tt_ids[i]
        eeids.append(id)
        results[id] = p


    # for w in range(1, 10):
    #     print 'week: %d' % w
    #     eids = train[train['week'] == w]['enrollment_id']
    #     tr = train[train['enrollment_id'].isin(eids)]
    #     tr = tr.drop(['enrollment_id', 'week'], axis=1)
    #
    #     tr_y = tr.dropout.values
    #
    #     pos_sum = np.sum(tr_y == 1)
    #     neg_sum = np.sum(tr_y == 0)
    #     tr_x = tr.drop('dropout', axis=1)
    #     dates[w] = (tr_x, tr_y)
    #     m, n = tr_x.shape
    #     print  'M: %d, N: %d, POS: %d; NEG: %d (%f)' % (m, n, pos_sum, neg_sum, neg_sum / (pos_sum + neg_sum + 0.0))
    #
    #     clf = create_clf('xgb', None)
    #     models[w] = clf
    #     auc = cv_loop(tr_x, tr_y, models[w], 5)
    #     print 'AUC (w=%d): %f' % (w, auc)
    #

    #     clf.fit(tr_x, tr_y)
    #
    #     tt_x = test[test['week'] == w]
    #     ids = tt_x.enrollment_id.values
    #     tt_x = tt_x.drop(['enrollment_id', 'week'], axis=1)
    #     preds = clf.predict_proba(tt_x)[:,1]
    #     for i, p in enumerate(preds):
    #         id = int(ids[i].split('-')[0])
    #         eeids.append(id)
    #         results[id] = p
    #
    eeids.sort()

    output = open("weekly_submission.csv", 'w')
    for k in eeids:
        p = results[k]
        # k = int(k.split('-')[0])
        # print '%d, %.6f' % (k, p)
        output.write('%d,%.6f\n' % (k, p))
    output.close()


if __name__ == '__main__':
    main()
