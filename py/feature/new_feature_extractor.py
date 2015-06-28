#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os

from new_feature_bag import NewFeatureBag
from feature_extractor import FeatureExtractor

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'

base_dir = os.path.dirname(__file__)

class NewFeatureExtractor(FeatureExtractor):
    def __init__(self, mode, data_type, log_csv_path, feature_path, debug_limit):
        FeatureExtractor.__init__(self, mode, data_type, log_csv_path, feature_path, debug_limit)

    def extract(self):
        tuple_iter = self._tuple_generator(self._filtered_iter)
        grouped_iter = itertools.groupby(tuple_iter, lambda x: x[0])
        bag_iter = self._bag_generator(grouped_iter)
        feature_iter = self._extract_simple_features(bag_iter)
        self._save_to_file(feature_iter)
        self._log_csv.close()

    def _extract_simple_features(self, iter):
        for bag in iter:
            print bag.enrollment_id
            yield bag.extract_request_count_lstday()\
                .extract_event_count_lstday()\
                .extract_lst_lag()\
                .extract_server_events_lstday()\
                .extract_problem_count_lstday()\
                .extract_video_count_lstday()\
                .extract_browser_events_lstday()\
                .extract_source_count_lstday()\
                .extract_hour_count_lstday()\
                .extract_hour_allocate()\
                .extract_hour_allocate_lstday()\
                .extract_session_count_2h()\
                .extract_session_count_lstday_2h()\
                .extract_session_count_3h()\
                .extract_session_count_lstday_3h()\
                .extract_session_lst2week_2h()\
                .extract_session_lst2week_3h()\
                .extract_session_lst1week_2h()\
                .extract_session_lst1week_3h()\
                .extract_session_per_day_2h()\
                .extract_session_per_day_3h()\
                .extract_staytime_lstday()\
                .extract_request_weekend_lstday()\
                .extract_daytime_lstday()\
                .extract_moduel_problem_lstday()\
                .extract_problem_over3minutes_count_lstday()\
                .extract_video_over10minutes_count_lstday()

    def _tuple_generator(self, iter):
        for line in iter:
            enrollment_id = line.split(',')[0]
            if str.isdigit(enrollment_id):
                yield (int(enrollment_id), self._parse_line(line))

    def _bag_generator(self, iter):
        for k, g in iter:
            yield NewFeatureBag(k, [t[1] for t in g], [], [])
