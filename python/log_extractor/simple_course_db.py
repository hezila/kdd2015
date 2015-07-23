#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os

from utils import *
from data import *

from feature_extractor import FeatureExtractor

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '19-06-2015'

base_dir = os.path.dirname(__file__)

class groupby(dict):
    def __init__(self, seq, key=lambda x:x):
        for value in seq:
            k = key(value)
            self.setdefault(k, []).append(value)
    __iter__ = dict.iteritems


module_cates = {
    "chapter",
    "problem",
    "sequential",
    "video"
    }


class SimpleCourseDB(FeatureExtractor):
    def __init__(self):
        train_log_path = base_dir + '/../../data/log_train.csv'
        test_log_path = base_dir + '/../../data/log_test.csv'
        label_path = base_dir + '/../../data/truth_train.csv'
        date_path = base_dir + '/../../data/date.csv'

        train_enrollment_path = base_dir + '/../../data/enrollment_train.csv'
        test_enrollment_path = base_dir + '/../../data/enrollment_test.csv'

        module_path = base_dir + '/../../data/object.csv'

        self.module_cates = module_cates

        labels = {}
        with open(label_path, 'r') as r:
            for line in r:
                eid, label = line.strip().split(',')
                if str.isdigit(eid):
                    labels[eid] = int(label)
        self.labels = labels

        course_by_eid = {}
        user_by_eid = {}

        courses_by_user = {}
        train_courses_by_user = {}
        test_courses_by_user = {}

        course_counts = {}
        train_course_counts = {}
        test_course_counts = {}

        course_drops = {}
        with open(train_enrollment_path, 'r') as r:
            for line in r:
                eid, uid, cid = line.strip().split(',')

                if str.isdigit(eid):
                    course_by_eid[eid] = cid
                    user_by_eid[eid] = uid
                    courses_by_user.setdefault(uid, []).append((eid, cid))
                    train_courses_by_user.setdefault(uid, []).append((eid, cid))

                    course_counts.setdefault(cid, 0.0)
                    course_counts[cid] += 1.0

                    train_course_counts.setdefault(cid, 0.0)
                    train_course_counts[cid] += 1.0

                    if eid in labels:
                        l = labels[eid]
                        if l == 1:
                            course_drops.setdefault(cid, 0.0)
                            course_drops[cid] += 1.0


        with open(test_enrollment_path, 'r') as r:
            for line in r:
                eid, uid, cid = line.strip().split(',')

                if str.isdigit(eid):
                    course_by_eid[eid] = cid
                    user_by_eid[eid] = uid
                    courses_by_user.setdefault(uid, []).append((eid, cid))
                    test_courses_by_user.setdefault(uid, []).append((eid, cid))

                    course_counts.setdefault(cid, 0.0)
                    course_counts[cid] += 1.0

                    test_course_counts.setdefault(cid, 0.0)
                    test_course_counts[cid] += 1.0

        self.course_by_eid = course_by_eid
        self.user_by_eid = user_by_eid
        self.courses_by_user = courses_by_user
        self.train_courses_by_user = train_courses_by_user
        self.test_courses_by_user = test_courses_by_user

        course_dates = {}
        with open(date_path, 'r') as r:
            i = 0
            for line in r:
                if i == 0:
                    i += 1
                    continue
                cid, start, end = line.strip().split(',')
                start = datetime.datetime.strptime(start, '%Y-%m-%d')
                end = datetime.datetime.strptime(end, '%Y-%m-%d')
                course_dates[cid] = (start, end)
        self.course_dates = course_dates

        train_user_drops = {}
        train_user_drop_ratio = {}
        for uid, items in courses_by_user.items():
            drops = 0.0
            for eid, cid in items:
                if not str.isdigit(eid):
                    continue

                if eid in labels:
                    l = labels[eid]

                    if l == 1:
                        drops += 1.0

            if drops > 0:
                drops = drops - 1.0
            train_user_drops[uid] = drops
            n = 0
            if uid in train_courses_by_user:
                n = len(train_courses_by_user[uid]) + 0.0
            train_user_drop_ratio[uid] = (drops + 8) / (n + 10)

        self.user_drops = train_user_drops
        self.user_drops_ratio = train_user_drop_ratio

        course_db = {}
        print 'course_size: %d' % len(course_drops)
        max_audience = 0
        for cid in course_counts.keys():
            all_audience = course_counts[cid]
            if all_audience > max_audience:
                max_audience = all_audience
            course_db[cid] = {'train_audience': train_course_counts[cid],
                              'test_audience': test_course_counts[cid],
                              'all_audience': all_audience,
                              'drops': course_drops[cid],
                              'drop_ratio': course_drops[cid] / (train_course_counts[cid] + 0.0)}

        for cid in course_counts.keys():
            course_pop = course_counts[cid] / max_audience
            course_db[cid]['course_pop'] = course_pop

        self.course_db = course_db

        self.module_db = load_modules(module_path)
        self.module_db.order_modules()

        self.time_modules = {}
        self.time_videos = {}

        self._train_log_csv = open(train_log_path, 'r')
        self._test_log_csv = open(test_log_path, 'r')

        self._filtered_iter = self._mode_filter(self._train_log_csv, self._test_log_csv)

    def _mode_filter(self, iter1, iter2, mode="normal"):
        for cnt, line in enumerate(iter1):
            # mode check
            if mode == 'debug' and cnt > self._debug_limit:
                break
            # remove invalid data (includes header)
            enrollment_id = (line.split(','))[0]
            if str.isdigit(enrollment_id):
                yield line

        for cnt, line in enumerate(iter2):
            # mode check
            if mode == 'debug' and cnt > self._debug_limit:
                break
            # remove invalid data (includes header)
            enrollment_id = (line.split(','))[0]
            if str.isdigit(enrollment_id):
                yield line
        self._train_log_csv.close()
        self._test_log_csv.close()


    def _parse_line(self, line):
        items = line.rstrip().split(',')

        dic = {
            'enrollment_id': items[0],
            'user_name': items[1],
            'course_id': items[2],
            'time': datetime.datetime.strptime(items[3], '%Y-%m-%dT%H:%M:%S'),
            'source': items[4],
            'event': items[5],
            'object': items[6]
        }
        return dic


    def build(self):
        tuple_iter = list(self._tuple_generator(self._filtered_iter))

        gp_dict = {}
        for k, value in tuple_iter:
            # k = key(value)
            gp_dict.setdefault(k, []).append(value)

        grouped_iter = [(k, v) for k, v in gp_dict.items()]

        self.cal_info(grouped_iter)

    def cal_info(self, iter):
        for cid, logs in iter:
            requests = sorted([(log['time'].strftime('%Y%m%d'), log['time']) for log in logs], key=lambda x: x[0])
            course_startday, start = requests[0]
            course_endday, end = requests[-1]
            course_duration = (end - start).days
            self.course_db[cid]['course_startday'] = datetime.datetime.strptime(course_startday, '%Y%m%d')
            self.course_db[cid]['course_endday'] = datetime.datetime.strptime(course_endday, '%Y%m%d')
            self.course_db[cid]['course_duration'] = course_duration

            counts = {}
            max_cnt = 0.0
            min_cnt = 100000000
            for d, t in requests:
                counts.setdefault(d, []).append(t)
            for d, v in counts.items():
                cnt = len(v) + 0.0
                if cnt > max_cnt:
                    max_cnt = cnt
                if cnt < min_cnt:
                    min_cnt = cnt
                counts[d] = cnt
            self.course_db[cid]['day_visits'] = counts
            self.course_db[cid]['day_maxvisit'] = max_cnt
            self.course_db[cid]['day_minvisit'] = min_cnt

            scale_counts = {}
            for d, c in counts.items():
                scale_counts[d] = c / max_cnt
            self.course_db[cid]['scale_visits'] = scale_counts

            for log in logs:
                if log['event'] == 'access':
                    t = log['time']
                    o = log['object']
                    self.time_modules.setdefault(o, []).append(t)
                if log['event'] == 'video':
                    t = log['time']
                    o = log['object']
                    self.time_videos.setdefault(o, []).append(t)
        print 'total modules: %d' % len(self.time_modules)
        print 'total videos: %d' % len(self.time_videos)

    def _tuple_generator(self, iter):
        results = []
        for line in iter:
            enrollment_id = line.split(',')[0]
            # uid, cid = self.enrollments[enrollment_id]
            cid = line.split(',')[2]
            if str.isdigit(enrollment_id):
                results.append((cid, self._parse_line(line)))
        return results

    def _bag_generator(self, iter):
        for k, g in iter:
            yield (k, g)
