#!/usr/bin/env python
#-*- coding: utf-8 -*-



__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '04-06-2015'

from collections import Counter
import datetime
import numpy as np
import os

from feature_bag import FeatureBag

base_dir = os.path.dirname(__file__)
event_types = ['problem', 'video', 'access', 'wiki', 'discussion', 'nagivate', 'page_close']

class EnrollmentFeatureBag(FeatureBag):
    def __init__(self, enrollment_id, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)

    def extract_request_count(self, skip_events=[]):
        self.feature_keys.append("request_count")
        if len(skip_events) == 0:
            self.feature_values.append(len(self.logs))
        else:
            x = sum(1 for log in self.logs if log['event'] not in skip_events)
            self.feature_values.append(x)
        return self

    def extract_request_days(self):
        request_dates = set([log['time'].strftime('%Y-%m-%d') for log in self.logs])
        self.feature_keys.append('request_days')
        self.feature_values.append(len(request_dates))
        return self
