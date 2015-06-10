#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os

from course_feature_bag import CourseFeatureBag
from feature_extractor import FeatureExtractor
from utils import *

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'

base_dir = os.path.dirname(__file__)

class CourseFeatureExtractor(FeatureExtractor):
    def __init__(self, mode, data_type, log_csv_path, enrollment_path, label_path, feature_path, debug_limit):
        labels = {}
        with open(label_path, 'r') as r:
            for line in r:
                eid, label = line.strip().split(',')
                if str.isdigit(eid):
                    labels[eid] = int(label)

        course_counts = {}
        course_drops = {}
        with open(enrollment_path, 'r') as r:
            for line in r:
                eid, uid, cid = line.strip().split(',')
                if str.isdigit(eid):
                    if cid not in course_counts:
                        course_counts[cid] = 1.0
                    else:
                        course_counts[cid] += 1.0

                    l = labels[eid]
                    if l == 1:
                        if cid not in course_drops:
                            course_drops[cid] = 1.0
                        else:
                            course_drops[cid] += 1.0


        course_db = {}
        for cid in course_counts.keys():
            course_db[cid] = (course_counts[cid], course_drops[cid])

        self.course_db = course_db



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
            yield bag.extract_course_audience(self.course_db)\
                .extract_course_drop(self.course_db)\
                .extract_course_dropratio(self.course_db)


    def _tuple_generator(self, iter):
        for line in iter:
            enrollment_id = line.split(',')[0]
            if str.isdigit(enrollment_id):
                yield (int(enrollment_id), self._parse_line(line))

    def _bag_generator(self, iter):
        for k, g in iter:
            yield CourseFeatureBag(k, [t[1] for t in g], [], [])
