#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os

from simple_feature_bag import SimpleFeatureBag
from feature_extractor import FeatureExtractor

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'

base_dir = os.path.dirname(__file__)

class SimpleFeatureExtractor(FeatureExtractor):
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
            yield bag.extract_request_year()\
                .extract_request_day()\
                .extract_duration_days()\
                .extract_request_count()\
                .extract_request_count_lst2week()\
                .extract_request_count_lst1week()\
                .extract_event_count()\
                .extract_event_count_lst2week()\
                .extract_event_count_lst1week()\
                .extract_active_days()\
                .extract_active_days_per_week()\
                .extract_active_days_lst2week()\
                .extract_active_days_lst1week()\
                .extract_server_events()\
                .extract_server_events_lst2week()\
                .extract_server_events_lst1week()\
                .extract_problem_count()\
                .extract_problem_count_lst2week()\
                .extract_video_count()\
                .extract_video_count_lst2week()\
                .extract_access_count()\
                .extract_access_count_lst2week()\
                .extract_access_count_lst1week()\
                .extract_browser_events()\
                .extract_browser_events_lst2week()\
                .extract_browser_events_lst1week()\
                .extract_source_count()\
                .extract_hour_count()\
                .extract_session_count()\
                .extract_session_lst2week()\
                .extract_session_lst1week()\
                .extract_session_per_day()\
                .extract_request_lag_min()\
                .extract_request_lag_max()\
                .extract_request_lag_mean()\
                .extract_request_lag_std()\
                .extract_request_lags()\
                .extract_staytime()\
                .extract_request_weekend_count()\
                .extract_request_weekend_percentage()\
                .extract_daytime()\
                .extract_moduel_problem()\
                .extract_problem_over3minutes_count()\
                .extract_video_over10minutes_count()

    def _tuple_generator(self, iter):
        for line in iter:
            enrollment_id = line.split(',')[0]
            if str.isdigit(enrollment_id):
                yield (int(enrollment_id), self._parse_line(line))

    def _bag_generator(self, iter):
        for k, g in iter:
            yield SimpleFeatureBag(k, [t[1] for t in g], [], [])
