#!/usr/bin/env python
#-*- coding: utf-8 -*-

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '04-06-2015'


class FeatureBag():
    def __init__(self, enrollment_id, logs, feature_keys, feature_values):
        self.enrollment_id = enrollment_id
        self.feature_keys = feature_keys
        self.feature_values = feature_values
        self.logs = logs
