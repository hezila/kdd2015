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

    def extract_duration_days(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        duration = (request_dates[-1] - request_dates[0]).days
        if duration == 0:
            duration = 1.0
        self.feature_keys.append('duration')
        self.feature_values.append(duration)
        return self

    def extract_request_count(self, skip_events=[]):
        self.feature_keys.append("request_count")
        if len(skip_events) == 0:
            self.feature_values.append(len(self.logs))
        else:
            x = sum(1 for log in self.logs if log['event'] not in skip_events)
            self.feature_values.append(x)
        return self

    def extract_active_days(self):
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs])
        self.feature_keys.append('active_days')
        self.feature_values.append(len(request_dates))
        return self

    def extract_active_days_per_week(self):
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs])
        active_days = len(request_dates)
        request_dates = sorted(list(request_dates))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        duration = (request_dates[-1] - request_dates[0]).days
        if duration == 0:
            duration = 1.0
        weeks = int(duration / 7.0)
        if weeks == 0:
            weeks = 1
        avg = active_days / (weeks + 0.0)
        self.feature_keys.append('active_days_per_week')
        self.feature_values.append(avg)
        return self

    def extract_fst_day(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        fst_day = request_dates[0]
        self.feature_keys.append('fst_day')
        self.feature_values.append(fst_day)
        return self
    
    def extract_lst_day(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        lst_day = request_dates[-1]
        self.feature_keys.append('lst_day')
        self.feature_values.append(lst_day)
        return self
