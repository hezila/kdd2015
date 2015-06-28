#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os

from course_feature_bag import CourseFeatureBag
from feature_extractor import FeatureExtractor
from simple_course_db import SimpleCourseDB
from utils import *
from data import *

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'

base_dir = os.path.dirname(__file__)



class CourseFeatureExtractor(FeatureExtractor):
    def __init__(self, mode, data_type, log_csv_path, enrollment_path, label_path, module_path, feature_path, debug_limit):
        self.db = SimpleCourseDB(mode, data_type, log_csv_path, enrollment_path, label_path, module_path, feature_path, debug_limit)
        self.db.build()
        print 'finish build course DB!'
        log_csv_path = base_dir + '/../../data/log_train.csv'
        FeatureExtractor.__init__(self, mode, data_type, log_csv_path, feature_path, debug_limit)


    def extract(self):
        tuple_iter = self._tuple_generator(self._filtered_iter)
        grouped_iter = itertools.groupby(tuple_iter, lambda x: x[0])
        bag_iter = self._bag_generator(grouped_iter)
        feature_iter = self._extract_enrollment_features(bag_iter)
        self._save_to_file(feature_iter)
        self._log_csv.close()

    def _extract_enrollment_features(self, iter):
        for bag in iter:
            yield bag.extract_course_audience(self.db)\
                .extract_left_module_count(self.db)\
                .extract_module_count(self.db)\
                .extract_module_lag2(self.db)\
                .extract_module_lag(self.db)\
                .extract_course_finish(self.db)\
                .extract_lag_nextmodule(self.db)\
                .extract_lag_lastmodule(self.db)\
                .extract_course_timeslot(self.db)\
                .extract_user_variables(self.db)


    def _tuple_generator(self, iter):
        for line in iter:
            enrollment_id = line.split(',')[0]
            if str.isdigit(enrollment_id):
                yield (int(enrollment_id), self._parse_line(line))

    def _bag_generator(self, iter):
        for k, g in iter:
            yield CourseFeatureBag(k, [t[1] for t in g], [], [])
