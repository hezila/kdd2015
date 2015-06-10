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
