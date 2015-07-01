#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from base_classifier import BaseClassifier

def preprocess_data(X, scaler=None):
    if not scaler:
        sclaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def process_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

class DeepNNClassifier(BaseClassifier):
    def __init__(self, neuro_num=512, nb_epoch=200, optimizer = "adam", scaler=None, verbose=False):
        self.neuro_num = neuro_num
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.scaler = scaler
        self.verbose = verbose

    def fit(self, X, y):
        y = np_utils.to_categorical(y.astype(np.int32))

        X, _ = preprocess_data(X, self.scaler)

        nb_classes = y.shape[1]

        dims = X.shape[1]

        neuro_num = self.neuro_num
        model = Sequential()
        model.add(Dense(dims, neuro_num, init='glorot_uniform'))
        model.add(PReLU((neuro_num,)))
        model.add(BatchNormalization((neuro_num,)))
        model.add(Dropout(0.5))

        model.add(Dense(neuro_num, neuro_num, init='glorot_uniform'))
        model.add(PReLU((neuro_num,)))
        model.add(BatchNormalization((neuro_num,)))
        model.add(Dropout(0.5))

        model.add(Dense(neuro_num, neuro_num, init='glorot_uniform'))
        model.add(PReLU((neuro_num,)))
        model.add(BatchNormalization((neuro_num,)))
        model.add(Dropout(0.5))

        model.add(Dense(neuro_num, nb_classes, init='glorot_uniform'))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)


        model.fit(X, y, nb_epoch=self.nb_epoch,
                  batch_size=256, validation_split=0.15,
                  show_accuracy=True,
                  shuffle=True)
        self.model = model

        self.num_class = len(np.unique(y))
        return self

    def predict_proba(self, X):
        X, _ = preprocess_data(X, self.scaler)
        preds = self.model.predict_proba(X)

        return preds
