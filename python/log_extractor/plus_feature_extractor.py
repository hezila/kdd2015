#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os

from plus_feature_bag import PlusFeatureBag
from feature_extractor import FeatureExtractor
from simple_course_db import SimpleCourseDB
from utils import *
from data import *


__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'

base_dir = os.path.dirname(__file__)



class PlusFeatureExtractor(FeatureExtractor):
    def __init__(self, mode, data_type, log_csv_path, feature_path, debug_limit):
        # self.db = SimpleCourseDB()
        # self.db.build()
        # print 'finish build course DB!'
        FeatureExtractor.__init__(self, mode, data_type, log_csv_path, feature_path, debug_limit)


    def extract(self):
        tuple_iter = self._tuple_generator(self._filtered_iter)
        grouped_iter = itertools.groupby(tuple_iter, lambda x: x[0])
        bag_iter = self._bag_generator(grouped_iter)
        feature_iter = self._extract_enrollment_features(bag_iter)
        self._save_to_file(feature_iter)
        self._log_csv.close()

    def _extract_enrollment_features(self, iter):
        i = 0
        for bag in iter:
            print '%d - %s' % (i, bag.enrollment_id)
            i += 1
            # yield bag.extract_user_features(self.db)\
            #     .extract_course_features(self.db)\
            #     .extract_visit_features(self.db)\
            #     .extract_moduletime_features(self.db)\
            #     .extract_module_features(self.db)\
            #     .extract_coursetime_features(self.db)
            yield bag.extract_azure_feature()


    def _tuple_generator(self, iter):
        for line in iter:
            enrollment_id = line.split(',')[0]
            if str.isdigit(enrollment_id):
                yield (enrollment_id, self._parse_line(line))

    def _bag_generator(self, iter):
        for k, g in iter:
            yield PlusFeatureBag(k, [t[1] for t in g], [], [])
