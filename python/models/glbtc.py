#!/usr/bin/env python
#-*- coding: utf-8 -*-

import graphlab as gl
import numpy as np
import pandas as pd

from base_classifier import BaseClassifier

class BoostedTreesClassifier(BaseClassifier):
    """
    Wrapper of Graph Lab Boosted Trees Classifier

    Parameters
    ----------
    max_iterations : int, optional
        The maximum number of iterations for boosting. Each iteration results
        in the creation of an extra tree.

    max_depth : float, optional
        Maximum depth of a tree.

    class_weights : {dict, auto}, optional
        Weights the examples in the training data according to the given class
        weights. If set to None, all classes are supposed to have weight one.
        The auto mode set the class weight to be inversely proportional to
        number of examples in the training data with the given class.

    step_size : float, [0,1], optional
        Step size (shrinkage) used in update to prevents overfitting. It shrinks
        the prediction of each weak learner to make the boosting process more
        conservative. The smaller the step size, the more conservative the
        algorithm will be. Smaller step_size work well when max_iterations is
        large.

    min_loss_reduction : float, optional
        Minimum loss reduction required to make a further partition on a leaf
        node of the tree. The larger it is, the more conservative the algorithm
        will be.

    min_child_weight : float, optional
        This controls the minimum number of instances needed for each leaf. The
        larger it is, the more conservative the algorithm will be. Set it larger
        when you want to prevent overfitting. Formally, this is minimum sum of
        instance weight (hessian) in each leaf. If the tree partition step
        results in a leaf node with the sum of instance weight less than
        min_child_weight, then the building process will give up further
        partitioning. For a regression task, this simply corresponds to minimum
        number of instances needed to be in each node.

    row_subsample : float, optional
        Subsample the ratio of the training set in each iteration of tree
        construction. This is called the bagging trick and can usually help
        prevent overfitting. Setting this to a value of 0.5 results in the model
        randomly sampling half of the examples (rows) to grow each tree.

    column_subsample : float, optional
        Subsample ratio of the columns in each iteration of tree construction.
        Like row_subsample, this can also help prevent model overfitting.
        Setting this to a value of 0.5 results in the model randomly sampling
        half of the columns to grow each tree.

    validation_set : SFrame, optional
        A dataset for monitoring the model's generalization performance. For
        each row of the progress table, the chosen metrics are computed for both
        the provided training dataset and the validation_set. The format of this
        SFrame must be the same as the training set. By default this argument is
        set to 'auto' and a validation set is automatically sampled and used for
        progress printing. If validation_set is set to None, then no additional
        metrics are computed. This is computed once per full iteration. Large
        differences in model accuracy between the training data and validation
        data is indicative of overfitting. The default value is 'auto'.

    verbose : boolean, optional
        Print progress information during training (if set to true).

    """

    def __init__(self, min_loss_reduction = 0,      class_weights = None,
                       step_size          = 0.3, min_child_weight = 0.1,
                       column_subsample   = 1,      row_subsample = 1,
                       max_depth          = 6,     max_iterations = 10,
                       verbose            = False):
        self.min_loss_reduction = min_loss_reduction
        self.class_weights      = class_weights
        self.step_size          = step_size
        self.min_child_weight   = min_child_weight
        self.column_subsample   = column_subsample
        self.row_subsample      = row_subsample
        self.max_depth          = max_depth
        self.max_iterations     = max_iterations
        self.verbose            = verbose

    def fit(self, X, y):
        """
        Fit the model according to the given training data

        Parameters
        ----------
        X: {array-like}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features

        y: array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            return self.
        """

        X = gl.SFrame(pd.DataFrame(X))
        X['target'] = y
        self.model = gl.boosted_trees_classifier.create(
                X, target = 'target', validation_set = None,
                min_loss_reduction  = self.min_loss_reduction,
                class_weights       = self.class_weights,
                step_size           = self.step_size,
                min_child_weight    = self.min_child_weight,
                column_subsample    = self.column_subsample,
                row_subsample       = self.row_subsample,
                max_depth           = self.max_depth,
                max_iterations      = self.max_iterations,
                verbose             = self.verbose)
        self.num_class = len(np.unique(y))

    def predict_proba(self, X):
        """
        Probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model
        """

        X = gl.SFrame(pd.DataFrame(X))
        preds = self.model.predict_topk(X, output_type = 'probability',
                                   k = self.num_class)
        preds['id'] = preds['id'].astype(int) + 1
        preds = preds.unstack(['class', 'probability'], 'probs').unpack(
                                'probs', '')
        preds = preds.sort('id')
        del preds['id']
        return preds.to_dataframe().as_matrix()

    def predict(self, X):
        """
        Class estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        yhat: array-like, shape = (n_samples, )
            Returns the predicted class of the sample
        """
        X = gl.SFrame(pd.DataFrame(X))
        yhat = self.model.predict(X)
        return np.array(yhat)

    def evaluate(self, X, y):
        X = gl.SFrame(pd.DataFrame(X))
        X['target'] = y
        return self.model.evaluate(X, metric='accuracy')

    def print_coefficients(self, num_rows=18):
        pass

if __name__ == "__main__":
    X = np.random.randn(10,3)
    y = np.random.randint(0, 2, 10)
    clf = BoostedTreesClassifier()
    clf.fit(X, y)
    yhat = clf.predict(X)

    p = clf.predict_proba(np.random.randn(1, 3))
    print p
    print p[0][1]
