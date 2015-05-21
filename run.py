#/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score

from dataset import *
from model.glbtc import BoostedTreesClassifier
from model.gllgc import LogisticClassifier
from model.sksgd import StochasticGradientClassifier

from optparse import OptionParser

def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)
    parser.add_option("-t", "--train", dest="train",
        help="the train file")
    parser.add_option("-s", "--test", dest="test", help="the test file")
    parser.add_option("-o", "--output", dest="output", help="the output file")
    # parser.add_option("-d", "--dataset", dest="dataset", help="the dataset file")
    # parser.add_option("-i", "--ids", dest="ids", help="the ids file")


    (options, remainder) = parser.parse_args()

    train_dataset, train_labels, test_dataset = load_dataset(options.train, options.test)

    # labels = encode_labels(train_labels)
    # print 'The labels samples: ',
    # print labels[:10]

    tr_x, tr_y, va_x, va_y = random_split(train_dataset, train_labels, 0.2)

    # params = {'target': 'target',
    #       'max_iterations': 250,
    #       'max_depth': 10,
    #       'min_child_weight': 4,
    #       'row_subsample': .9,
    #       'min_loss_reduction': 1,
    #       'column_subsample': .8,
    #       'validation_set': None}



    # clf = BoostedTreesClassifier(min_loss_reduction = 1.0,       class_weights = None,
    #                                 step_size       = .1,    min_child_weight = 100,
    #                                 column_subsample= 1,   row_subsample    = .6,
    #                                 max_depth       = 3,     max_iterations  = 80)


    clf = LogisticClassifier()

    # clf = StochasticGradientClassifier(alpha = 0.01, n_iter = 150)

    clf.fit(tr_x, tr_y)
    # coefficients = clf.model['coefficients']
    # print coefficients

    print clf.evaluate(va_x, va_y)

    yhat = clf.predict_proba(va_x)

    preds = yhat[:,1]

    fpr, tpr, thresholds = roc_curve(va_y, preds)
    roc_auc = auc(fpr, tpr)
    print "AUC: %f" % (roc_auc)

    sys.exit(0)

    # clf.fit(train_dataset.values, train_labels)

    ids = test_dataset.eid.values
    test_dataset = test_dataset.drop('eid', axis=1)

    yhat = clf.predict_proba(test_dataset.values)
    preds = yhat[:,1]

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
