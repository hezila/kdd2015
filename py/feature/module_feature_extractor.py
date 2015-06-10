#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os
import datetime

from module_feature_bag import ModuleFeatureBag
from feature_extractor import FeatureExtractor
from utils import *
from data import *

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-09-2015'

base_dir = os.path.dirname(__file__)


class ModuleFeatureExtractor(FeatureExtractor):
    def __init__(self, mode, data_type, log_csv_path, module_path, feature_path, debug_limit):

        self.module_db = load_modules(module_path)
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
            yield bag.extract_module_freq_mean(self.module_db)\
                    .extract_module_freq_median(self.module_db)\
                    .extract_module_freq_max(self.module_db)\
                    .extract_module_fstaclag_mean(self.module_db)\
                    .extract_module_fstaclag_median(self.module_db)\
                    .extract_module_fstaclag_min(self.module_db)\
                    .extract_module_fstaclag_max(self.module_db)\
                    .extract_module_fstaclag_25p(self.module_db)\
                    .extract_module_fstaclag_75p(self.module_db)\
                    .extract_module_lstaclag_mean(self.module_db)\
                    .extract_module_lstaclag_median(self.module_db)\
                    .extract_module_lstaclag_min(self.module_db)\
                    .extract_module_lstaclag_max(self.module_db)\
                    .extract_module_lstaclag_25p(self.module_db)\
                    .extract_module_lstaclag_75p(self.module_db)\
                    .extract_module_tau(self.module_db)\
                    .extract_module_backjumps(self.module_db)



    def _tuple_generator(self, iter):
        for line in iter:
            enrollment_id = line.split(',')[0]
            if str.isdigit(enrollment_id):
                yield (int(enrollment_id), self._parse_line(line))

    def _bag_generator(self, iter):
        for k, g in iter:
            yield ModuleFeatureBag(k, [t[1] for t in g], [], [])
