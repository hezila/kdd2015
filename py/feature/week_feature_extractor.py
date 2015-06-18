#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os, datetime

from week_feature_bag import WeekFeatureBag
from feature_extractor import FeatureExtractor

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'

base_dir = os.path.dirname(__file__)

class WeekFeatureExtractor(FeatureExtractor):
    def __init__(self, mode, data_type, log_csv_path, feature_path, label_path, debug_limit):
        FeatureExtractor.__init__(self, mode, data_type, log_csv_path, feature_path, debug_limit)
        labels = {}
        with open(label_path, 'r') as r:
            for line in r:
                eid, dropout = line.strip().split(',')
                if str.isdigit(eid):
                    labels[int(eid)] = int(dropout)
        self.labels = labels


    def extract(self):
        tuple_iter = self._tuple_generator(self._filtered_iter)
        grouped_iter = itertools.groupby(tuple_iter, lambda x: x[0])
        bag_iter = self._bag_generator(grouped_iter)
        feature_iter = self._extract_enrollment_features(bag_iter)
        self._save_to_file(feature_iter)
        self._log_csv.close()

    def _extract_enrollment_features(self, iter):
        for bag in iter:
            for sub_bag in bag.gen_sub_bags():
                yield sub_bag.extract_duration_days()\
                    .extract_request_count()\
                    .extract_active_days()\
                    .extract_avg_active_days()\
                    .extract_request_lag_min()\
                    .extract_request_lag_max()\
                    .extract_request_lag_mean()\
                    .extract_request_lag_var()\
                    .extract_request_lag_3days()\
                    .extract_request_lag_3days_ratio()\
                    .extract_request_lag_5days()\
                    .extract_request_lag_5days_ratio()\
                    .extract_request_lag_1week()\
                    .extract_request_lag_1week_ratio()\
                    .extract_request_hours()\
                    .extract_request_hour_count()\
                    .extract_request_hour_mean()\
                    .extract_request_hour_var()\
                    .extract_request_weekend_count()\
                    .extract_request_weekend_percentage()\
                    .extract_session_mean()\
                    .extract_session_var()\
                    .extract_staytime_min()\
                    .extract_staytime_max()\
                    .extract_staytime_mean()\
                    .extract_staytime_var()\
                    .extract_server_event_count()\
                    .extract_browser_event_count()\
                    .extract_event_count()\
                    .extract_event_percentage()\
                    .extract_daytime()\
                    .extract_video_over10minutes_count()\
                    .extract_video_over10minutes_count_lst1week()\
                    .extract_problem_over3minutes_count()\
                    .extract_problem_over3minutes_count_lst1week()\
                    .extract_request_count_lst1weeks()\
                    .extract_server_event_count_lst1weeks()\
                    .extract_browser_event_count_lst1weeks()\
                    .extract_activedays_lst1weeks()\
                    .extract_source_count()

                    # .extract_event_days_per_week()\
                    # .extract_video_over10minutes_count_per_week()\
                    # .extract_problem_over3minutes_count_per_week()\

    def _tuple_generator(self, iter):
        for line in iter:
            enrollment_id = line.split(',')[0]
            if str.isdigit(enrollment_id):
                yield (int(enrollment_id), self._parse_line(line))

    def _bag_generator(self, iter):
        for k, g in iter:
            label = None
            if self._data_type == 'train':
                label = self.labels[k]
            yield WeekFeatureBag(k, label, None, [t[1] for t in g], [], [])

    def _save_to_file(self, iter):
        output = None
        if self._data_type == 'train':
            output = open('week_train_truth.csv', 'w')
        with open(self._feature_path, 'w') as feature_file:
            header_written = False
            for bag in iter:
                if header_written is not True:
                    if output:
                        output.write('enrollment_id,dropout\n')
                    feature_file.write('enrollment_id,{0}\n'.format(
                            ','.join(bag.feature_keys)
                    ))
                    header_written = True
                if output:
                    output.write('{0},{1}\n'.format(str(bag.enrollment_id), bag.label))
                feature_file.write('{0},{1}\n'.format(
                        str(bag.enrollment_id),
                        ','.join([str(v) for v in bag.feature_values])
                ))


            feature_file.close()
            if output:
                output.close()
