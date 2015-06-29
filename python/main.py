#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from util import *
from dataset import *
from blend import *

def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)

    parser.add_option("-t", "--task", dest="task", help="the task name")
    parser.add_option("-o", "--output", dest="output", help="the output file")

    (options, remainder) = parser.parse_args()

    train_paths = [
        "../data/train_simple_feature.csv"
        ]
    label_path = "../data/truth_train.csv"

    test_paths = [
        "../data/test_simple_feature.csv"
        ]

    train = merge_features(train_paths, label_path)
    y = encode_labels(train.dropout.values)
    train = train.drop('dropout', axis=1)
    X = train.drop('enrollment_id', axis=1)
    m, n = X.shape
    print 'train.shape=%s' % (str(X.shape))


    test = merge_features(test_paths)
    tt_ids = test.enrollment_id.values
    X_test = test.drop('enrollment_id', axis=1)
    print 'test.shape=%s' % (str(X_test.shape))

    scaler = StandardScaler().fit(np.vstack((X, X_test)))

    task = options.task
    if not task:
        task = "blend"

    if task == 'blend':

        clf_list = [
            ("extra_gini_10depth", create_clf("ext", {"criterion": "gini", "n_estimators": 200, "max_depth": 10})),
            ("extra_entropy_10depth", create_clf("ext", {"criterion": "entropy", "n_estimators": 200, "max_depth": 10})),
            ("extra_gini_20depth", create_clf("ext", {"criterion": "gini", "n_estimators": 200, "max_depth": 20})),
            ("extra_entropy_20depth", create_clf("ext", {"criterion": "entropy", "n_estimators": 200, "max_depth": 20})),
            ("extra_gini_30depth", create_clf("ext", {"criterion": "gini", "n_estimators": 200, "max_depth": 30})),
            ("extra_entropy_30depth", create_clf("ext", {"criterion": "entropy", "n_estimators": 200, "max_depth": 30})),
            ("rfc_gini_3depth", create_clf("rfc", {"criterion": "gini", "max_depth": 3, "n_estimators": 200})),
            ("rfc_entropy_3depth", create_clf("rfc", {"criterion": "entropy", "max_depth": 3, "n_estimators": 200})),
            ("rfc_gini_5depth", create_clf("rfc", {"criterion": "gini", "max_depth": 5, "n_estimators": 200})),
            ("rfc_entropy_5depth", create_clf("rfc", {"criterion": "entropy", "max_depth": 5, "n_estimators": 200})),
            ("rfc_gini_6depth", create_clf("rfc", {"criterion": "gini", "max_depth": 6, "n_estimators": 200})),
            ("rfc_entropy_6depth", create_clf("rfc", {"criterion": "entropy", "max_depth": 6, "n_estimators": 200})),
            ("rfc_gini_8depth", create_clf("rfc", {"criterion": "gini", "max_depth": 8, "n_estimators": 200})),
            ("rfc_entropy_8depth", create_clf("rfc", {"criterion": "entropy", "max_depth": 8, "n_estimators": 200})),
            ("rfc_gini_10depth", create_clf("rfc", {"criterion": "gini", "max_depth": 10, "n_estimators": 200})),
            ("rfc_entropy_10depth", create_clf("rfc", {"criterion": "entropy", "max_depth": 10, "n_estimators": 200})),
            ("rfc_gini_12depth", create_clf("rfc", {"criterion": "gini", "max_depth": 12, "n_estimators": 200})),
            ("rfc_entropy_12depth", create_clf("rfc", {"criterion": "entropy", "max_depth": 12, "n_estimators": 200})),
            ("xgb_1500_2depth", create_clf("xgb", {"max_depth": 2, "n_estimators": 1500, "learning_rate": 0.03})),
            ("xgb_600_3depth", create_clf("xgb", {"max_depth": 3, "n_estimators": 600, "learning_rate": 0.03})),
            ("xgb_600_4depth", create_clf("xgb", {"max_depth": 4, "n_estimators": 600, "learning_rate": 0.03})),
            ("xgb_600_5depth", create_clf("xgb", {"max_depth": 5, "n_estimators": 600, "learning_rate": 0.03})),
            ("xgb_600_6depth", create_clf("xgb", {"max_depth": 6, "n_estimators": 600, "learning_rate": 0.02})),
            ("xgb_600_7depth", create_clf("xgb", {"max_depth": 7, "n_estimators": 600, "learning_rate": 0.01})),
            ("xgb_600_8depth", create_clf("xgb", {"max_depth": 8, "n_estimators": 600, "learning_rate": 0.01})),
            ("lgc_1c_scale", create_clf("lgc", {"C": 1.0, "scaler": scaler})),
            ("lgc_1c", create_clf("lgc", {"C": 1.0})),
            ("lgc_3c_scale", create_clf("lgc", {"C": 3.0, "scaler": scaler})),
            ("lgc_3c", create_clf("lgc", {"C": 3.0})),
            ("lgc_5c_scale", create_clf("lgc", {"C": 5.0, "scaler": scaler})),
            ("lgc_5c", create_clf("lgc", {"C": 5.0})),
            ]
        
        X = X.values
        blend_train, blend_test = train_blend(X, y, X_test, clf_list, 5)

        #blender = create_clf('lgc', {"C": 1.0})
        blender = create_clf('ext', {"max_depth": 8, "criterion": "entropy", "n_estimator": 100})
        
        # blender = blender.fit(blend_train, y)
        auc = cv_loop(blend_train, y, blender)
        print "AUC (all): %f" % auc
    elif task == 'lgc':
        print 'Try logistic regression ..'
        clf = create_clf("lgc", {"C": 1})
        auc = cv_loop(X, y, clf, 5)
        print 'AUC (all): %f' % auc

    elif task == "ext":
        print 'Try ExtraTreeClassifier'
        #clf = create_clf("ext", {"max_depth": 10}) # 0.86261
        #clf = create_clf("ext", {"max_depth": 20}) # 0.862636
        #clf = create_clf("ext", {"max_depth": 30}) # 0.860944
        #clf = create_clf("ext", {"criterion": "entropy", "max_depth": 10}) # 0.862610
        #clf = create_clf("ext", {"criterion": "entropy", "max_depth": 20}) # 0.862564
        clf = create_clf("ext", {"criterion": "entropy", "max_depth": 20, "n_estimators": 100}) # 0.861795
        #clf = create_clf("ext", {"criterion": "entropy", "max_depth": 20, "n_estimators": 2000}) # 0.862695
        #clf = create_clf("ext", {"criterion": "entropy", "max_depth": 30, "n_estimators": 2000}) # 0.860
        auc = cv_loop(X, y, clf, 5)
        print 'AUC (all): %f' % auc

    elif task == 'rfc':
        print 'Try RFC ..'
        #clf = create_clf('rfc', {'max_depth': 5}) # 0.859583
        #clf = create_clf("rfc", {"criterion": "entropy", "max_depth": 10}) # 0.863369
        #clf = create_clf("rfc", {"criterion": "entropy", "max_depth": 10, "n_estimators": 200}) # 0.863285
        # clf = create_clf("rfc", {"criterion": "entropy", "max_depth": 10, "n_estimators": 100}) # 0.863207
        #clf = create_clf("rfc", {"criterion": "entropy", "max_depth": 10, "max_features": None, "n_estimators": 200}) # 0.863341
        #clf = create_clf("rfc", {"criterion": "gini", "max_depth": 10, "max_features": None, "n_estimators": 200}) # 0.863291
        auc = cv_loop(X, y, clf, 5)
        print 'AUC (all): %f' % auc

    elif task == "xgb":
        #clf = create_clf('xgb', {"max_depth": 2, "n_estimators": 1500, "learning_rate": 0.03}) # 0.860279
        clf = create_clf('xgb', {"max_depth": 8, "n_estimators": 600, "learning_rate": 0.01})
        auc = cv_loop(X, y, clf, 5)
        print "AUC (all): %f" % auc

if __name__ == '__main__':
    main()
        
