#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import Counter
import datetime
import os
import math

from feature_bag import FeatureBag
from data import *
# import scipy.stats

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-09-2015'


class WeekModuleFeatureBag(FeatureBag):
    def __init__(self, enrollment_id, week, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)
        self.logs = sorted(logs, key=lambda x: x['time'])
        self.start = self.logs[0]['time'].strftime('%Y%m%d')
        self.start_day = datetime.datetime.strptime(self.start, '%Y%m%d')
        self.end = self.logs[-1]['time'].strftime('%Y%m%d')
        self.end_day = datetime.datetime.strptime(self.end, '%Y%m%d')
        if week:
            self.week = week
        else:
            self.week = int((self.end_day - self.start_day).days / 7)

    def gen_sub_bags(self):

        start_weekday = self.start_day.weekday()
        next_monday = self.start_day + datetime.timedelta(days=8 - start_weekday)

        w = 1
        for i in range(len(self.logs)):
            log = self.logs[i]
            if log['time'] > next_monday:
                bag_logs = self.logs[:i]
                next_monday = next_monday + datetime.timedelta(days=7)
                lag = (log['time'] - self.logs[i-1]['time']).days
                if lag > 9:
                    l = 1
                else:
                    l = 0
                sub_bag = WeekModuleFeatureBag('{0}-{1}w'.format(self.enrollment_id, w), w, bag_logs, [], [])
                w += 1

                print '{0}-{1}w-{2}'.format(self.enrollment_id, w, l)
                yield sub_bag


        # w = (self.end_day - self.start_day).days + 1
        # w = int(w / 7)
        # if w == 0:
        #     w = 1

        print '{0}-{1}w'.format(self.enrollment_id, w)
        yield WeekModuleFeatureBag('{0}-{1}w'.format(self.enrollment_id, w), w, self.logs, [], [])


    def extract_module_freq_mean(self, module_db):
        mids = [log['object'] for log in self.logs if log['event'] == 'access']
        counter = Counter(mids)
        counts = []
        for mid in counter:
            counts.append(counter[mid])
        if len(counts) == 0:
            mean = 1.0
        else:
            mean = sum(counts) / (len(counts) + 0.0)
        self.feature_keys.append('module_freq_mean')
        self.feature_values.append(mean)
        return self

    def extract_module_freq_median(self, module_db):
        mids = [log['object'] for log in self.logs if log['event'] == 'access']
        counter = Counter(mids)
        counts = []
        for mid in counter:
            counts.append(counter[mid])
        if len(counts) == 0:
            m = 1.0
        else:
            m = median(counts)
        self.feature_keys.append('module_freq_median')
        self.feature_values.append(m)
        return self

    def extract_module_freq_max(self, module_db):
        mids = [log['object'] for log in self.logs if log['event'] == 'access']
        counter = Counter(mids)
        counts = []
        for mid in counter:
            counts.append(counter[mid])
        if len(counts) == 0:
            m = 1.0
        else:
            m = max(counts)
        self.feature_keys.append('module_freq_max')
        self.feature_values.append(m)
        return self

    def extract_module_fstaclag_mean(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t < at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]

        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60

        mean = 7
        if len(lags) > 0:
            mean = sum(lags) / (len(lags) + 0.0)
        self.feature_keys.append("module_fstaclag_mean")
        self.feature_values.append(mean)
        return self

    def extract_module_lstaclag_mean(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t > at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60

        mean = 7
        if len(lags) > 0:
            mean = sum(lags) / (len(lags) + 0.0)
        self.feature_keys.append("module_lstaclag_mean")
        self.feature_values.append(mean)
        return self

    def extract_module_fstaclag_median(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t < at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = median(lags)
        self.feature_keys.append("module_fstaclag_median")
        self.feature_values.append(m)
        return self

    def extract_module_lstaclag_median(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t > at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = median(lags)
        self.feature_keys.append("module_lstaclag_median")
        self.feature_values.append(m)
        return self


    def extract_module_fstaclag_min(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t < at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = min(lags)
        self.feature_keys.append("module_fstaclag_min")
        self.feature_values.append(m)
        return self

    def extract_module_lstaclag_min(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t > at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = min(lags)
        self.feature_keys.append("module_lstaclag_min")
        self.feature_values.append(m)
        return self

    def extract_module_fstaclag_max(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t < at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = max(lags)
        self.feature_keys.append("module_fstaclag_max")
        self.feature_values.append(m)
        return self

    def extract_module_lstaclag_max(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t > at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = max(lags)
        self.feature_keys.append("module_lstaclag_max")
        self.feature_values.append(m)
        return self

    def extract_module_fstaclag_25p(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t < at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = percentile(lags, 0.25)
        self.feature_keys.append("module_fstaclag_25p")
        self.feature_values.append(m)
        return self

    def extract_module_lstaclag_25p(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t > at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = percentile(lags, 0.25)
        self.feature_keys.append("module_lstaclag_25p")
        self.feature_values.append(m)
        return self

    def extract_module_fstaclag_75p(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t < at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = percentile(lags, 0.75)
        self.feature_keys.append("module_fstaclag_75p")
        self.feature_values.append(m)
        return self

    def extract_module_lstaclag_75p(self, module_db):
        mids = [(log['object'], log['time']) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t > at:
                    mid_times[mid] = (t, st)
        lags = [(t - s).days for t, s in mid_times.values()]
        for i, d in enumerate(lags):
            if d < -7:
                lags[i] = -7.0
            elif d > 60:
                lags[i] = 60
        m = 7
        if len(lags) > 0:
            m = percentile(lags, 0.75)
        self.feature_keys.append("module_lstaclag_75p")
        self.feature_values.append(m)
        return self


    def extract_module_tau(self, module_db):
        mids = [(log['object'], log['time'].strftime('%Y%m%d%H%M')) for log in self.logs if log['event'] == 'access']
        mid_times = {}
        for mid, t in mids:
            if mid not in mid_times:
                s = module_db.get_start(mid)
                if s:
                    s = s.strftime('%Y%m%d%H%M')
                    mid_times[mid] = (t, s)
            else:
                at, st = mid_times[mid]
                if t < at:
                    mid_times[mid] = (t, st)

        release_times = []
        access_times = []
        for mid in mid_times.keys():
            at, st = mid_times[mid]
            access_times.append(int('{0}'.format(at)))
            release_times.append(int('{0}'.format(st)))

        tau = 0.0
        if len(release_times) > 1:
            tau = scipy.stats.kendalltau(release_times, access_times)[0]
        if math.isnan(tau):
            tau = 0.0
        self.feature_keys.append("module_tau")
        self.feature_values.append(tau)
        return self

    def extract_module_backjumps(self, module_db):
        mids = sorted([(log['object'], log['time']) for log in self.logs if log['event'] == 'access'], key=lambda x: x[1])
        release_times = {}
        for mid, at in mids:
            s = module_db.get_start(mid)
            if s is not None:
                release_times[mid] = s
        orders = order_dict(release_times)
        last_r = 1
        cnt = 0.0
        for mid, at in mids:
            if mid not in orders:
                continue
            r = orders.index(mid) + 1
            if r < last_r:
                cnt += 1.0
            last_r = r
        self.feature_keys.append("module_backjumps")
        self.feature_values.append(cnt)
        return self
