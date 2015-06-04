#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os

from enrollment_feature_bag import EnrollmentFeatureBag
from feature_extractor import FeatureExtractor

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'

base_dir = os.path.dirname(__file__)

class EnrollmentFeatureExtractor(FeatureExtractor):
    def __init__(self, mode, data_type, log_csv_path, feature_path, debug_limit):
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
            yield bag.extract_duration_days()\
                .extract_request_count()\
                .extract_active_days()\
                .extract_active_days_per_week()\
                .extract_fst_day()\
                .extract_lst_day()

    def _tuple_generator(self, iter):
        for line in iter:
            enrollment_id = line.split(',')[0]
            if str.isdigit(enrollment_id):
                yield (int(enrollment_id), self._parse_line(line))

    def _bag_generator(self, iter):
        for k, g in iter:
            yield EnrollmentFeatureBag(k, [t[1] for t in g], [], [])
