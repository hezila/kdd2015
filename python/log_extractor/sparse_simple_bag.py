#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import Counter
import datetime
import time
import os
import math

from feature_bag import FeatureBag
from utils import *

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-18-2015'

base_dir = os.path.dirname(__file__)
event_types = ['problem', 'video', 'access', 'wiki', 'discussion', 'nagivate', 'page_close']
server_events = ['access', 'wiki', 'discussion', 'problem', 'nagivate']
browser_events = ['access', 'video', 'problem', 'page_close']

class SparseSimpleFeatureBag(FeatureBag):
    def __init__(self, enrollment_id, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)
        self.logs = sorted(logs, key=lambda x: x['time'])
        self.start = self.logs[0]['time'].strftime('%Y%m%d')
        self.start_day = datetime.datetime.strptime(self.start, '%Y%m%d')

        self.end = self.logs[-1]['time'].strftime('%Y%m%d')
        self.end_day = datetime.datetime.strptime(self.end, '%Y%m%d')

    def extract_request_year(self):
        y = int(self.start_day.strftime('%Y%m'))
        if y <= 201402:
            self.feature_keys.append('year_2013')
            self.feature_values.append(1)
            self.feature_keys.append('year_2014')
            self.feature_values.append(0)
        else:
            self.feature_keys.append('year_2013')
            self.feature_values.append(0)
            self.feature_keys.append('year_2014')
            self.feature_values.append(1)
        return self

    def extract_request_mean_time(self):
        sum_time = reduce(lambda x, y: x+y, [time.mktime(log['time'].timetuple()) for log in self.logs])
        mean_time = sum_time / (len(self.logs) + 0.0)
        mean_time = datetime.datetime.fromtimestamp(mean_time)
        m = int(mean_time.strftime('%m'))
        d = int(mean_time.strftime('%d'))
        # print 'm: %d - d: %d' % (m, d)
        for i in range(1, 13):
            self.feature_keys.append('month_%d' % i)
            v = 0
            if m == i:
                v = 1
            self.feature_values.append(v)
        for i in range(1, 31):
            self.feature_keys.append('day_%d'% i)
            v = 0
            if m == i:
                v = 1
            elif m >= 30 and i == 30:
                v = 1
            self.feature_values.append(v)
        return self

    def extract_request_exam_holiday(self):
        fst_m = int(self.start_day.strftime('%m'))

        if fst_m in [7, 1]:
            self.feature_keys.append('exam_time_fstday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('exam_time_fstday')
            self.feature_values.append(0)

        if fst_m in [7, 8, 2]:
            self.feature_keys.append('holiday_fstday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('holiday_fstday')
            self.feature_values.append(0)

        lst_m = int(self.end_day.strftime('%m'))
        if lst_m in [7, 1]:
            self.feature_keys.append('exam_time_lstday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('exam_time_lstday')
            self.feature_values.append(0)

        if lst_m in [7, 8, 2]:
            self.feature_keys.append('holiday_lstday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('holiday_lstday')
            self.feature_values.append(0)

        if fst_m in [7, 1] and lst_m in [7, 1]:
            self.feature_keys.append('in_exam')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('in_exam')
            self.feature_values.append(0)

        if fst_m in [7, 8, 2] and lst_m in [7, 8, 2]:
            self.feature_keys.append('in_holiday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('in_holiday')
            self.feature_values.append(0)

        return self

    def extract_duration_days(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        duration = (request_dates[-1] - request_dates[0]).days
        if duration == 0:
            duration = 1.0

        if duration <= 2:
            self.feature_keys.append('duration_less2day')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('duration_less2day')
            self.feature_values.append(0)

        if duration <= 7 and duration > 2:
            self.feature_keys.append('duration_less1week')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('duration_less1week')
            self.feature_values.append(0)

        if duration > 7 and duration <= 14:
            self.feature_keys.append('duration_2week')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('duration_2week')
            self.feature_values.append(0)

        if duration >= 15:
            self.feature_keys.append('duration_over2weeks')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('duration_over2weeks')
            self.feature_values.append(0)
        return self


    def extract_request_lags(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        lags = []
        if len(request_dates) >= 2:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        sum_lag = sum(lags) + 0.0
        self.feature_keys.append('total_lag')
        self.feature_values.append(sum_lag + .0)

        x2d = 0.0
        x2d_ratio = 0.0
        x3d = 0.0
        x3d_ratio = 0.0
        x5d = 0.0
        x5d_ratio = 0.0
        x7d = 0.0
        x7d_ratio = 0.0

        for lag in lags:
            if lag <= 2:
                x2d += 1.0
            if lag >= 3 and lag < 5:
                x3d += 1.0
            if lag >= 5 and lag < 8:
                x5d += 1.0
            if lag >= 8:
                x7d += 1.0

        if len(lags) > 0:
            n = len(lags) + 0.0
            x2d_ratio = x2d / n
            x3d_ratio = x3d / n
            x5d_ratio = x5d / n
            x7d_ratio = x7d / n


        self.feature_keys.append('sumlag<1week')
        self.feature_keys.append('sumlag1<>2week')
        self.feature_keys.append('sumlag>2week')
        if sum_lag == 0:
            self.feature_values.append(1)
            self.feature_values.append(1)
            self.feature_values.append(1)
        elif sum_lag < 7:
            self.feature_values.append(1)
            self.feature_values.append(0)
            self.feature_values.append(0)
        elif sum_lag >= 7 and sum_lag < 14:
            self.feature_values.append(0)
            self.feature_values.append(1)
            self.feature_values.append(0)
        else:
            self.feature_values.append(0)
            self.feature_values.append(0)
            self.feature_values.append(1)

        return self



    def extract_request_weekend_count(self):
        ws = [log['time'].weekday() for log in self.logs]
        d = len([1 for w in ws if w > 5]) + 0.0

        if d > (len(ws) - d):
            self.feature_keys.append('more_weekend')
            self.feature_values.append(1)
            self.feature_keys.append('more_workday')
            self.feature_values.append(0)
        else:
            self.feature_keys.append('more_weekend')
            self.feature_values.append(0)
            self.feature_keys.append('more_workday')
            self.feature_values.append(1)

        return self


    def extract_daytime(self):
        request_dates = [int(log['time'].strftime('%H')) for log in self.logs]
        counts = {}
        day_times = 0.0
        night_times = 0.0
        for h in request_dates:
            # if h not in counts:
            #     counts[h] = 1.0
            # else:
            #     counts[h] += 1.0
            if h < 19 and h >= 7:
                day_times += 1.0
            else:
                night_times += 1.0



        if night_times >= day_times:
            self.feature_keys.append('day_time')
            self.feature_values.append(0)

            self.feature_keys.append('night_time')
            self.feature_values.append(1)

        else:
            self.feature_keys.append('day_time')
            self.feature_values.append(1)

            self.feature_keys.append('night_time')
            self.feature_values.append(0)
        return self
