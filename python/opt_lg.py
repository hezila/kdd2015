import numpy
from numpy.random import multivariate_normal
rng = numpy.random

import theano
import theano.tensor as T

import optunity
import optunity.metrics

from util import *
from dataset import *

###################################
# Load data set
###################################

train_paths = [
    "./blend_train.csv"
    ]

label_path = "../data/truth_train.csv"

test_paths = [
    "./blend_test.csv"
    ]

train = merge_features(train_paths, label_path)
labels = encode_labels(train.dropout.values)
train = train.drop(['dropout','enrollment_id'], axis=1)

####################################
# LR regression training function
####################################


training_steps = 2000

def train_lr(x_train, y_train, regularization=0.01, step=0.1):
    x = T.matrix("x")
    y = T.vector("y")
    w = theano.shared(rng.randn(feats), name="w")
    b = theano.shared(0., name="b")

    # Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))                 # Probability that target = 1
    prediction = p_1
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)           # Cross-entropy loss function
    cost = xent.mean() + regularization * (w ** 2).sum()    # The cost to minimize
    gw, gb = T.grad(cost, [w, b])                           # Compute the gradient of the cost
                                                            # (we shall return to this in a
                                                            # following section of this tutorial)

    # Compile
    train = theano.function(
            inputs=[x,y],
            outputs=[prediction, xent],
            updates=((w, w - step * gw), (b, b - step * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)

    # Train
    for i in range(training_steps):
        train(x_train, y_train)
    return predict, w, b


#############################
# LR evaluation functions
#############################
def lr_untuned(x_train, y_train, x_test, y_test):
    predict, w, b = train_lr(x_train, y_train)
    yhat = predict(x_test)
    loss = optunity.metrics.logloss(y_test, yhat)
    brier = optunity.metrics.brier(y_test, yhat)
    return loss, brier

def lr_tuned(x_train, y_train, x_test, y_test):
    @optunity.cross_validated(x=x_train, y=y_train, num_folds=3)
    def inner_cv(x_train, y_train, x_test, y_test, regularization, step):
        predict, _, _ = train_lr(x_train, y_train,
                                regularization=regularization, step=step)
        yhat = predict(x_test)
        return optunity.metrics.logloss(y_test, yhat)

    pars, _, _ = optunity.minimize(inner_cv, num_evals=50,
                                regularization=[0.001, 0.05],
                                step=[0.01, 0.2])
    predict, w, b = train_lr(x_train, y_train, **pars)
    yhat = predict(x_test)
    loss = optunity.metrics.logloss(y_test, yhat)
    brier = optunity.metrics.brier(y_test, yhat)
    return loss, brier

# wrap both evaluation functions in cross-validation
# we will compute two metrics using nested cross-validation
# for this purpose we use list_mean() as aggregator
outer_cv = optunity.cross_validated(x=train, y=labels, num_folds=3,
                                    aggregator=optunity.cross_validation.list_mean)
lr_untuned = outer_cv(lr_untuned)
lr_tuned = outer_cv(lr_tuned)

print('true model: 1 + 2 * x1 + 3 * x2')
print('')

# perform experiment
print('evaluating untuned LR model')
untuned_loss, untuned_brier = lr_untuned()
