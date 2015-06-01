#/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score

from dataset import *
from util import *
from model.glbtc import BoostedTreesClassifier
from model.gllgc import LogisticClassifier
from model.glknn import KNNClassifier

from model.sksgd import StochasticGradientClassifier

from optparse import OptionParser


# List of candidate family classifiers with parameters
# [name, classifier object].
def candidate_families():
    candidates = []

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
                                    column_subsample= 0.5,   row_subsample    = .6,
                                    max_depth       = 5,     max_iterations  = 120)
    candidates.append(["GBT", gbtcf])

    return candidates

def cal_auc(y, preds):
    fpr, tpr, thresholds = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def xval(classifier, dataset, labels, n_folds = 3):
    log_info('Crossvalidation started... ')

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
        # log_info('Quality of split... ' + str(quality))
        print 'AUC in fold %d: %f' % (i, roc_auc)
    quality = avg_quality / n_folds
    # log_info('Estimated quality of model... ' + str(quality))
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


def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)
    parser.add_option("-t", "--train", dest="train",
        help="the train file")
    parser.add_option("-s", "--test", dest="test", help="the test file")
    parser.add_option("-o", "--output", dest="output", help="the output file")
    parser.add_option("-n", "--nfolds", dest="nfolds", help="the folds number")
    # parser.add_option("-d", "--dataset", dest="dataset", help="the dataset file")
    # parser.add_option("-i", "--ids", dest="ids", help="the ids file")


    (options, remainder) = parser.parse_args()

    train_dataset, train_labels, test_dataset = load_dataset(options.train, options.test,
                                                                with_norm = True,
                                                                verbose = True)

    # labels = encode_labels(train_labels)
    # print 'The labels samples: ',
    # print labels[:10]

    cols = train_dataset.columns

    log_featurs = [col for col in cols if col not in ['#daytime_ratio', '#day_time', '#night_time' '#weekend_ratio', '#weekend_time', '#weekday_time', '#lagging2week', '#lagging<2week', '#ratio_browser']]
    log_transf_replace(train_dataset, log_featurs)
    log_transf_replace(test_dataset, log_featurs)

    norm_transformer = NormTransformer(cols, exludes = ['#day_time', '#night_time', '#weekend_time', '#weekday_time', '#lagging2week', '#lagging<2week',])

    norm_transformer.fit_transform(train_dataset)
    norm_transformer.transform(test_dataset)

    # filters = ['wiki_times_log', 'wiki_times_invert_log',
    #             'problem_times_log',
    #             'nagivate_times_invert_log',
    #             'discussion_times_invert_log',
    #             'duration_invert_log',
    #             'active_days_invert_log',
    #             'video_times_log',
    #             'access_times_invert_log'
    #             ]

    filters = []
    fs = FeatureSelector()

    nfolds = 5
    if options.nfolds:
        nfolds = int(options.nfolds)


    # td = fs.drop_transform(train_dataset, filters)

    # tr_x, tr_y, va_x, va_y = random_split(train_dataset, train_labels)

    cds = candidate_families()

    # ensembel_fit(cds, tr_x, tr_y)
    weights = {"SGD": 0.1, "LGC": 0.15, "GBT": 0.75}
    # preds = ensembel(cds, va_x, va_y, weights)



    ### Feature select ###

    # clf = BoostedTreesClassifier(min_loss_reduction = 0.5,       class_weights = None,
    #                                 step_size       = .1,    min_child_weight = 50,
    #                                 column_subsample= 0.6,   row_subsample    = .6,
    #                                 max_depth       = 5,     max_iterations  = 100)

    # clf = LogisticClassifier(l2_penalty = 0.1,
    #                             max_iterations = 250,
    #                             step_size = 0.2,
    #                             convergence_threshold = 0.001,
    #                             verbose = False)

    # d = None
    # aucs = {}
    # if len(filters) > 0:
    #     d = fs.drop_transform(train_dataset, filters)
    #     aucs['all'] = xval(clf, d, train_labels, nfolds)
    # else:
    #     aucs['all'] = xval(clf, train_dataset, train_labels, nfolds)
    #
    #
    # sys.exit(0)
    #
    #
    # for c in cols:
    #     if c in filters:
    #         continue
    #
    #     print "Eliminate feature: %s" % c
    #     d = fs.drop_transform(train_dataset, [c] + filters)
    #     aucs[c] = xval(clf, d, train_labels, nfolds)
    # orders = order_dict(aucs)[::-1]
    #
    # print 'Filters: %s' % ' '.join(filters)
    # for c in orders:
    #     print '%s: %f (%f)' % (c, aucs[c], aucs[c] - aucs['all'])
    #

    # sys.exit(0)

    ensembel_fit(cds, train_dataset.values, train_labels)

    ###   predict for test dataset ###

    # clf.fit(train_dataset.values, train_labels)

    ids = test_dataset.eid.values
    test_dataset = test_dataset.drop('eid', axis=1)
    # test = fs.drop_transform(test_dataset, filters)

    # yhat = clf.predict_proba(test_dataset.values)
    # preds = yhat[:,1]
    #
    preds = ensembel(cds, test_dataset.values, None, weights)


    results = {}
    eids = []
    for i, p in enumerate(preds):
        id = int(ids[i])
        eids.append(id)
        results[id] = p

    eids.sort()

    output = open(options.output, 'w')
    for k in eids:
        p = results[k]
        print '%d, %.6f' % (k, p)
        output.write('%d,%.6f\n' % (k, p))
    output.close()

    # output = open(options.output, 'w')
    # for i, p in enumerate(preds):
    #     x = test_dataset.values[i]
    #     # print x
    #     x = np.asmatrix(x)
    #     p2 = clf.predict_proba(x)[0][1]
    #     print '%s, %.6f, %.6f' % (ids[i], p, p2)
    #     output.write('%s,%.6f\n' % (ids[i], p))
    # output.close()


    # print roc_auc_score(np.array([l for l in va_y]), np.array([p for p in yhat[:1]]))

if __name__ == '__main__':
    main()
