#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import Counter
import datetime
import os

from feature_bag import FeatureBag

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '08-06-2015'


class CourseFeatureBag(FeatureBag):
    def __init__(self, enrollment_id, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)
        self.logs = sorted(logs, key=lambda x: x['time'])

    def extract_course_audience(self, db):
        cid = self.logs[0]['course_id']
        audience = db.course_db[cid]['audience']
        drops = db.course_db[cid]['drops']
        self.feature_keys.append('course_audience')
        self.feature_values.append(audience)

        self.feature_keys.append('course_drop')
        self.feature_values.append(drops)

        self.feature_keys.append('course_dropratio')
        self.feature_values.append(drops / (audience + 0.0))

        return self

    def extract_left_module_count(self, db):
        module_db = db.module_db
        cid = self.logs[0]['course_id']
        course_size = len(module_db.modules_by_cid[cid])

        last_r = 1
        for log in self.logs:
            mid = log['object']
            r = module_db.get_rank(cid, mid)
            if r is not None:
                if r > last_r:
                    last_r = r

        l = course_size - last_r
        self.feature_keys.append('left_module_count')
        self.feature_values.append(l)
        return self

    def extract_module_count(self, db):
        module_db = db.module_db
        cid = self.logs[0]['course_id']
        course_size = len(module_db.modules_by_cid[cid])
        self.feature_keys.append('course_module_count')
        self.feature_values.append(course_size)

        m_size = len(set([log['object'] for log in self.logs if (log['event'] == 'access' and module_db.exist(log['object']))]))

        self.feature_keys.append('module_count')
        self.feature_values.append(m_size)

        ratio = m_size / (course_size + 0.0)
        self.feature_keys.append('module_ratio')
        self.feature_values.append(ratio)

        return self

    def extract_course_finish(self, db):
        module_db = db.module_db
        cid = self.logs[0]['course_id']
        course_size = len(module_db.modules_by_cid[cid])

        last_r = 1
        for log in self.logs:
            mid = log['object']
            r = module_db.get_rank(cid, mid)
            if r is not None:
                if r > last_r:
                    last_r = r

        l = 1.0 - (course_size - last_r) / (course_size + 0.0)
        self.feature_keys.append('course_finish_ratio')
        self.feature_values.append(l)
        return self

    def extract_module_lag2(self, db):
        time_modules = db.time_modules
        module_db = db.module_db
        cid = self.logs[0]['course_id']

        modules = module_db.modules_by_cid[cid]

        access_times = {}
        for log in self.logs:
            o = log['object']
            if o in modules:
                access_times.setdefault(o, []).append(log['time'])

        for mid in access_times.keys():
            access_times[mid] = min(access_times[mid])

        lags = []
        for mid in access_times.keys():
            s = min(time_modules[mid])
            # e = max(time_modules[mid])
            t = access_times[mid]
            l = (t - s).days
            lags.append(l)

        if len(lags) == 0:
            self.feature_keys.append('access_module')
            self.feature_values.append(0)
            lags.append(300)
        else:
            self.feature_keys.append('access_module')
            self.feature_values.append(1)


        self.feature_keys.append('max_module_lag')
        self.feature_values.append(max(lags))

        self.feature_keys.append('min_module_lag')
        self.feature_values.append(min(lags))

        self.feature_keys.append('sum_module_lag')
        self.feature_values.append(sum(lags))

        self.feature_keys.append('mean_module_lag')
        self.feature_values.append(sum(lags) / (len(lags) + 0.0))

        return self

    def extract_module_lag(self, db):
        time_modules = db.time_modules
        # print 'LEN: %d' % len(time_modules)
        module_db = db.module_db
        cid = self.logs[0]['course_id']
        modules = module_db.modules_by_cid[cid]

        mids = [log for log in self.logs if log['event'] == 'access']
        mids = sorted([(log['object'], log['time']) for log in mids], key=lambda x: x[1])

        mids_keys = [m for m, t in mids]
        if len(mids_keys) > 0:
            m, last_time = mids[-1]
        else:
            last_time = self.logs[-1]['time']
        lags = []
        for mid in modules:
            if mid not in time_modules:
                continue

            if mid not in mids_keys:
                l = (min(time_modules[mid]) - last_time).days
                if l > 0:
                    lags.append(l)
        if len(lags) == 0:
            lags.append(0)

        self.feature_keys.append('nextmodule_lag')
        self.feature_values.append(min(lags))
        self.feature_keys.append('lastmodule_lag')
        self.feature_values.append(max(lags))
        return self

    def extract_lag_nextmodule(self, db):
        module_db = db.module_db
        cid = self.logs[0]['course_id']
        # course = course_db.get_course(cid)
        course_size = len(module_db.modules_by_cid[cid])

        last_r = 1
        last_time = None
        for log in self.logs:
            mid = log['object']
            r = module_db.get_rank(cid, mid)
            if r is not None:
                if r > last_r:
                    last_r = r
                    last_time = log['time']

        l = course_size - last_r
        d = 0
        if l > 0:
            next_m = module_db.modules_by_cid[cid][last_r]
            next_t = module_db.get_start(next_m)
            if next_t and last_time:
                d = (next_t - last_time).days
        if d < 0:
            d = 0

        self.feature_keys.append('lag_nextmodule')
        self.feature_values.append(d)
        return self

    def extract_lag_lastmodule(self, db):
        module_db = db.module_db
        cid = self.logs[0]['course_id']
        # course = course_db.get_course(cid)
        course_size = len(module_db.modules_by_cid[cid])

        last_r = 1
        last_time = None
        for log in self.logs:
            mid = log['object']

            r = module_db.get_rank(cid, mid)
            if r is not None:
                if r > last_r:
                    last_r = r
                    last_time = log['time']

        l = course_size - last_r
        d = 0
        last_m = module_db.modules_by_cid[cid][-1]
        last_t = module_db.get_start(last_m)
        # print last_time + "<<<<<"
        # print last_t + ">>>>"
        if last_t and last_time:
            d = (last_t - last_time).days
        if d < 0:
            d = 0

        self.feature_keys.append('lag_lastmodule')
        self.feature_values.append(d)
        return self

    def extract_course_timeslot(self, db):
        course_db = db.course_db
        cid = self.logs[0]['course_id']
        start_day = course_db[cid]['course_startday']
        end_day = course_db[cid]['course_endday']

        d = (end_day - start_day).days + 0.0

        # print 'cid-%s: %d' % (cid, d)
        days_after_course_start = (self.logs[0]['time'] - start_day).days
        if days_after_course_start < 0:
            days_after_course_start = 0
        self.feature_keys.append('days_after_course_start')
        self.feature_values.append(days_after_course_start)
        self.feature_keys.append('days_has_passed_ratio')
        self.feature_values.append(days_after_course_start / d)

        days_before_course_end = (end_day - self.logs[-1]['time']).days
        if days_before_course_end < 0:
            days_before_course_end = 0.0
        self.feature_keys.append('days_before_course_end')
        self.feature_values.append(days_before_course_end)

        self.feature_keys.append('days_before_course_ratio')
        self.feature_values.append(days_before_course_end / d)

        log_start = self.logs[0]['time'].strftime('%Y%m%d')
        # log_start = datetime.datetime(log_start, '%Y%m%d')

        visits = course_db[cid]['day_visits']
        scale_visits = course_db[cid]['scale_visits']

        v = 0.0
        sv = 0.0
        if log_start in visits:
            v = visits[log_start]
            sv = scale_visits[log_start]

        self.feature_keys.append('fst_day_pop')
        self.feature_values.append(v)

        self.feature_keys.append('fst_day_scale_pop')
        self.feature_values.append(sv)

        log_end = self.logs[-1]['time'].strftime('%Y%m%d')
        # log_end = datetime.datetime(log_end, '%Y%m%d')
        v = 0.0
        sv = 0.0
        if log_end in visits:
            v = visits[log_end]
            sv = scale_visits[log_end]
        self.feature_keys.append('lst_day_pop')
        self.feature_values.append(v)
        self.feature_keys.append('lst_day_scale_pop')
        self.feature_values.append(sv)

        return self



    def extract_user_variables(self, db):
        courses_by_user = db.courses_by_user
        uid = self.logs[0]['user_name']
        cnt = 0.0
        if uid in courses_by_user:
            cnt = len(courses_by_user[uid]) + 0.0

        self.feature_keys.append('user_courses')
        self.feature_values.append(cnt)

        user_drops_ratio = db.user_drops_ratio
        ratio = 0.7
        if uid in user_drops_ratio:
            ratio = user_drops_ratio[uid]

        self.feature_keys.append('user_drop_ratio')
        self.feature_values.append(ratio)
        return self
