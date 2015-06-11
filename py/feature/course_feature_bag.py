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

    def extract_course_audience(self, course_db):
        cid = self.logs[0]['course_id']
        audience, finish = course_db[cid]
        self.feature_keys.append('course_audience')
        self.feature_values.append(audience)
        return self

    def extract_course_drop(self, course_db):
        cid = self.logs[0]['course_id']
        audience, drops = course_db[cid]
        self.feature_keys.append('course_drop')
        self.feature_values.append(drops)

        return self

    def extract_course_dropratio(self, course_db):
        cid = self.logs[0]['course_id']
        audience, drops = course_db[cid]
        self.feature_keys.append('course_dropratio')
        self.feature_values.append(drops / (audience + 0.0))

        return self

    def extract_left_module_count(self, module_db):
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

    def extract_course_finish(self, module_db):
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

    def extract_lag_nextmodule(self, module_db):
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

    def extract_lag_lastmodule(self, module_db):
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
