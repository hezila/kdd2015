#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import Counter
import datetime
import os

from feature_bag import FeatureBag

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '04-06-2015'

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

    def extract_request_lag_min(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            lags = [7] # one week in this case
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        self.feature_keys.append('min_request_lag')
        self.feature_values.append(min(lags))
        return self

    def extract_request_lag_max(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            lags = [7]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        self.feature_keys.append('max_request_lag')
        self.feature_values.append(max(lags))
        return self

    
    def extract_request_lag_mean(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            lags = [7]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        self.feature_keys.append('mean_request_lag')
        self.feature_values.append(sum(lags) / (len(lags) + 0.0)) # avoid numpy to use pypy
        return self
    
    def extract_request_lag_var(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            var = 14
        elif len(request_dates) == 2:
            var = (request_dates[1] - request_dates[0]).days
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
            mean = sum(lags) / (len(lags) + 0.0)
            var = 1.0 / (len(lags) - 1) * sum([(l - mean)**2 for l in lags])
        self.feature_keys.append('var_request_lag')
        self.feature_values.append(var)
        return self

    
    def extract_request_hours(self):
        request_hours = sorted([log['time'].strftime('%H') for log in self.logs])
        counter = Counter(request_hours)
        for i in xrange(24):
            h = '{0:02d}'.format(i)
            cnt = 0
            if h in counter:
                cnt = counter[h]
            self.feature_keys.append('request_hour_{0}'.format(h))
            self.feature_values.append(cnt)
        return self

    # TODO
    # def extract_request_hours_00_05(self):
    #     return self

    def extract_request_hour_count(self):
        request_hours = set([int(log['time'].strftime('%H')) for log in self.logs])
        self.feature_keys.append('request_hour_count')
        self.feature_values.append(len(request_hours))
        return self

    def extract_request_hour_mean(self):
        request_hours = [int(log['time'].strftime('%H')) for log in self.logs]
        self.feature_keys.append('request_hour_mean')
        self.feature_values.append(sum(request_hours) / (len(request_hours) + 0.0))
        return self

    def extract_request_hour_var(self):
        request_hours = [int(log['time'].strftime('%H')) for log in self.logs]
        n = len(request_hours) + 0.0
        mean = sum(request_hours) / n

        var = 1.0 / (n - 1) * sum([(h - mean)**2 for h in request_hours]) if n > 1 else 12

        self.feature_keys.append('request_hour_var')
        self.feature_values.append(var)
        return self

    
    def extract_request_weekend_count(self):
        ws = [log['time'].weekday() for log in self.logs]
        self.feature_keys.append('request_weekend_count')
        self.feature_values.append(len([1 for w in ws if w > 5]))
        return self

    def extract_request_weekend_percentage(self):
        ws = [log['time'].weekday() for log in self.logs]
        self.feature_keys.append('request_weekend_percentage')
        self.feature_values.append(float(sum([1 for w in ws if w > 5])) / len(ws))
        return self
