#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd

from util import *
from dataset import *
from blend import *
from semi import *


def write_submission(tt_ids, preds, file_path):
    # preds = blender.predict_proba(blend_test)[:,1]
    # linear stretch of predictions to [0,1]
    # preds = (preds - preds.min())/(preds.max() - preds.min())
    eeids = []
    results = {}
    for i, p in enumerate(preds):
        id = tt_ids[i]
        eeids.append(id)
        results[id] = p

    eeids.sort()

    output = open(file_path, 'w')
    for k in eeids:
        p = results[k]
        output.write('%d,%f\n' % (k, p))
    output.close()

def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)

    parser.add_option("-t", "--task", dest="task", help="the task name")
    parser.add_option("-o", "--output", dest="output", help="the output file")

    (options, remainder) = parser.parse_args()

    train_paths = [
        "../data/train_simple_feature.csv",
        "../data/train_plus_feature.csv",
        "../data/train_azure_plus_feature.csv",
        #"../data/train_azure_feature.csv",
        #"../data/train_module_feature.csv",
        #"../data/train_course_feature.csv",
        #"./blend_train.csv"
        ]
    label_path = "../data/truth_train.csv"

    test_paths = [
        "../data/test_simple_feature.csv",
        "../data/test_plus_feature.csv",
        "../data/test_azure_plus_feature.csv",
        #"../data/test_azure_feature.csv",
        #"../data/test_module_feature.csv",
        #"../data/test_course_feature.csv",
        #"./blend_test.csv"
        ]

    train = merge_features(train_paths, label_path)
    train = train.drop(['user_drop_ratio'], axis=1)
    #train['user_drop_ratio'] = (train['user_drop_ratio'] + 8.0 / train['user_courses']) / (1.0 + 10.0 / train['user_courses'])
    y = encode_labels(train.dropout.values)
    train = train.drop('dropout', axis=1)
    tr_ids = train.enrollment_id.values
    X = train.drop('enrollment_id', axis=1)
    m, n = X.shape
    print 'train.shape=%s' % (str(X.shape))


    test = merge_features(test_paths)
    test = test.drop(['user_drop_ratio'], axis=1)
    #test['user_drop_ratio'] = (test['user_drop_ratio'] + 8.0 / test['user_courses']) / (1.0 + 10.0 / test['user_courses'])
    tt_ids = test.enrollment_id.values
    X_test = test.drop('enrollment_id', axis=1)
    print 'test.shape=%s' % (str(X_test.shape))

    scaler = StandardScaler().fit(np.vstack((X, X_test)))

    task = options.task
    if not task:
        task = "blend"

    if task == 'blend':

        clf_list = [
            #("knn_p2_10", create_clf('knn', {"n_neighbors": 10, "p": 2})),
            #("knn_p2_10_scaler", create_clf('knn', {"n_neighbors": 10, "p": 2, "scaler": scaler})),
            #("knn_p2_100", create_clf('knn', {"n_neighbors": 100, "p": 2})),
            #("knn_p2_100_scaler", create_clf('knn', {"n_neighbors": 100, "p": 2, "scaler": scaler})),
            #("knn_p2_500", create_clf('knn', {"n_neighbors": 500, "p": 2})),
            #("knn_p2_500_scaler", create_clf('knn', {"n_neighbors": 500, "p": 2, "scaler": scaler})),
            #("knn_p2_100", create_clf('knn', {"n_neighbors": 100, "p": 2})),
            #("knn_p2_800", create_clf('knn', {"n_neighbors": 800, "p": 2})),
            #("knn_p1_10", create_clf('knn', {"n_neighbors": 10, "p": 1})),
            #("knn_p1_10_scaler", create_clf('knn', {"n_neighbors": 10, "p": 1, "scaler": scaler})),
            #("knn_p1_100", create_clf('knn', {"n_neighbors": 100, "p": 1})),
            #("knn_p1_100_scaler", create_clf('knn', {"n_neighbors": 100, "p": 1, "scaler": scaler})),
            #("knn_p1_500", create_clf('knn', {"n_neighbors": 500, "p": 1})),
            #("knn_p1_500_scaler", create_clf('knn', {"n_neighbors": 500, "p": 1, "scaler": scaler})),
            #("knn_p1_800", create_clf('knn', {"n_neighbors": 800, "p": 1})),
            #("knn_p1_800_scaler", create_clf('knn', {"n_neighbors": 800, "p": 1, "scaler": scaler})),
            #("extra_gini_10depth", create_clf("ext", {"criterion": "gini", "n_estimators": 200, "max_depth": 10})),
            #("extra_entropy_10depth", create_clf("ext", {"criterion": "entropy", "n_estimators": 200, "max_depth": 10})),
            #("extra_gini_20depth", create_clf("ext", {"criterion": "gini", "n_estimators": 200, "max_depth": 20})),
            #("extra_entropy_20depth", create_clf("ext", {"criterion": "entropy", "n_estimators": 200, "max_depth": 20})),
            ("extra_gini_30depth", create_clf("ext", {"criterion": "gini", "n_estimators": 200, "max_depth": 30})),
            ("extra_entropy_30depth", create_clf("ext", {"criterion": "entropy", "n_estimators": 200, "max_depth": 30})),
            #("rfc_gini_3depth", create_clf("rfc", {"criterion": "gini", "max_depth": 3, "n_estimators": 200})),
            #("rfc_entropy_3depth", create_clf("rfc", {"criterion": "entropy", "max_depth": 3, "n_estimators": 200})),
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
            #("xgb_1500_2depth", create_clf("xgb", {"max_depth": 2, "n_estimators": 1500, "learning_rate": 0.03})),
            #("xgb_600_3depth", create_clf("xgb", {"max_depth": 3, "n_estimators": 600, "learning_rate": 0.03})),
            #("xgb_600_4depth", create_clf("xgb", {"max_depth": 4, "n_estimators": 600, "learning_rate": 0.03})),
            ("xgb_600_5depth", create_clf("xgb", {"max_depth": 5, "n_estimators": 600, "learning_rate": 0.03})),
            #("xgb_600_6depth", create_clf("xgb", {"max_depth": 6, "n_estimators": 600, "learning_rate": 0.02})),
            #("xgb_600_7depth", create_clf("xgb", {"max_depth": 7, "n_estimators": 600, "learning_rate": 0.01})),
            #("xgb_600_8depth", create_clf("xgb", {"max_depth": 8, "n_estimators": 600, "learning_rate": 0.01})),
            #("lgc_1c_scale", create_clf("lgc", {"C": 1.0, "scaler": scaler})),
            #("lgc_1c", create_clf("lgc", {"C": 1.0})),
            #("lgc_1c_l1", create_clf("lgc", {"C": 1.0, "penalty": "l1"})),
            #("lgc_3c_scale", create_clf("lgc", {"C": 3.0, "scaler": scaler})),
            #("lgc_3c", create_clf("lgc", {"C": 3.0})),
            ("lgc_3c_l1", create_clf("lgc", {"C": 3.0, "penalty": "l1"})),
            #("lgc_5c_scale", create_clf("lgc", {"C": 5.0, "scaler": scaler})),
            #("lgc_5c", create_clf("lgc", {"C": 5.0})),
            ]

        X = X.values
        blend_train, blend_test = train_blend(X, y, X_test, clf_list, 5)
        print 'blend_train.shape=%s' % (str(blend_train.shape))
        print 'blend_test.shape=%s' % (str(blend_test.shape))


        cols = [cname for cname, clf in clf_list]
        cols = ['enrollment_id'] + cols
        blend_train_ids = np.hstack((np.matrix(tr_ids).T, blend_train))
        blend_test_ids = np.hstack((np.matrix(tt_ids).T, blend_test))
        dump_data(blend_train_ids, cols, "new_blend_train.csv")
        dump_data(blend_test_ids, cols, "new_blend_test.csv")


        blender = create_clf('lgc', {"C": 1.0, "penalty": "l1"})
        auc = cv_loop(blend_train, y, blender)
        print 'AUC (LGC blend): %f' % auc

        blender = create_clf('ext', {"max_depth": 10, "criterion": "entropy", "n_estimator": 100})
        auc = cv_loop(blend_train, y, blender)
        print 'AUC (EXT blend): %f' % auc

        blender = create_clf('xgb', {'max_depth': 2, "n_estimators": 150, "learning_rate": 0.05})
        auc = cv_loop(blend_train, y, blender)
        print "AUC (XGB blend {d: %d, n: %d}): %f" % (2, 150, auc)
        blender = create_clf('xgb', {'max_depth': 3, "n_estimators": 200, "learning_rate": 0.05})
        auc = cv_loop(blend_train, y, blender)
        print 'AUC (XGB blend {d: %d, n: %d}): %f' % (3, 200, auc)

        blender = create_clf('xgb', {'max_depth': 3, "n_estimators": 100, "learning_rate": 0.1})
        blender = blender.fit(blend_train, y)
        preds = blender.predict_proba(blend_test)[:,1]
        write_submission(tt_ids, preds, "new_blend_submission.csv")

        combined_train = np.hstack((X, blend_train))
        combined_test = np.hstack((X_test, blend_test))
        blender = create_clf('xgb', {'max_depth': 5, "n_estimators": 600, "learning_rate": 0.03})
        blender = blender.fit(combined_train, y)
        preds = blender.predict_proba(combined_test)[:,1]
        write_submission(tt_ids, preds, "new_combined_blend_submission.csv")

    elif task == 'lgc':
        print 'Try logistic regression ..'
        clf = create_clf("lgc", {"C": 3, "scaler": scaler, "penalty": "l1"})
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
        clf = create_clf("rfc", {"criterion": "entropy", "max_depth": 40, "max_features": None, "n_estimators": 10000, "min_samples_split": 100}) # 0.863291
        auc = cv_loop(X, y, clf, 5)
        print 'AUC (all): %f' % auc

    elif task == 'knn':
        clf = create_clf('knn', {"n_neighbors": 800, "p": 2, "scaler": scaler})
        auc = cv_loop(X, y, clf, 5)
        print 'AUC (all): %f' % auc

    elif task == "gbt":
        paras = json.load(open('paras/gbt.json', 'r'))
        clf = create_clf("gbt", paras)
        clf = clf.fit(X, y)
        preds = clf.predict_proba(X_test)[:,1]
        write_submission(tt_ids, preds, "gbt_submission.csv")

    elif task == "xgb":
        #clf = create_clf('xgb', {"max_depth": 2, "n_estimators": 1500, "learning_rate": 0.03}) # 0.860279
        #clf = create_clf('xgb', {"max_depth": 5, "n_estimators": 600, "learning_rate": 0.03}) # public: 0.8891443712867697;
        clf = create_clf('xgb', {"max_depth": 5, "n_estimators": 600, "learning_rate": 0.03}) # public:
        auc = cv_loop(X, y, clf, 5)
        print "AUC (all): %f" % auc
        #sys.exit()
        clf = clf.fit(X, y)
        preds = clf.predict_proba(X_test)[:,1]
        write_submission(tt_ids, preds, 'xgb_new_submission.csv')
    elif task == "deep":
        clf = create_clf('deep', {"neuro_num": 512, "nb_epoch": 20, "scaler": scaler, "optimizer": "adadelta"})
        auc = cv_loop(X, y, clf, 5)
        print 'AUC (all): %f' % auc
        sys.exit(0)

        clf = clf.fit(X, y)
        preds = clf.predict_proba(X_test)[:,1]
        write_submission(tt_ids, preds, 'deep_submission.csv')

    elif task == 'semi':
        clf = create_clf("ext", {"criterion": "entropy", "max_depth": 20, "n_estimators": 100}) # 0.861795
        train_semi(X, y, X_test, clf, 5)

    elif task == 'gbc':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=300,
                                        learning_rate=0.1,
                                        min_samples_split=50,
                                        min_samples_leaf=50,
                                        max_depth=10,
                                        subsample=0.6,
                                        max_features='log2',
                                        verbose=1)
        auc = cv_loop(X, y, clf, 5)
        print 'AUC (all): %f' % auc

    elif task == 'label':
        from sklearn.semi_supervised import LabelPropagation
        from sklearn.semi_supervised import LabelSpreading
        label_prop_model = LabelSpreading()
        all_X = np.vstack((X, X_test))
        tm, tn = X_test.shape
        unlabeles = [-1] * tm
        ys = [list(y)]
        ys.append(unlabeles)
        labels = np.concatenate(ys)
        print 'ALL shape=%s' % (str(all_X.shape))
        print 'ALL y shape=%s' % (str(labels))
        label_prop_model.fit(all_X, labels)


if __name__ == '__main__':
    main()
