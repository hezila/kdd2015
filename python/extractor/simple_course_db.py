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


class SimpleCourseDB(FeatureExtractor):
    def __init__(self, mode, data_type, log_csv_path, enrollment_path, label_path, module_path, feature_path, debug_limit):
        train_log_path = base_dir + '/../../data/log_train.csv'
        test_log_path = base_dir + '/../../data/log_test.csv'
        label_path = base_dir + '/../../data/truth_train.csv'

        train_enrollment_path = base_dir + '/../../data/enrollment_train.csv'
        test_enrollment_path = base_dir + '/../../data/enrollment_test.csv'

        module_path = base_dir + '/../../data/object.csv'

        labels = {}
        with open(label_path, 'r') as r:
            for line in r:
                eid, label = line.strip().split(',')
                if str.isdigit(eid):
                    labels[eid] = int(label)
        self.labels = labels

        courses_by_user = {}
        # enrollments = {}
        course_counts = {}
        course_drops = {}
        with open(train_enrollment_path, 'r') as r:
            for line in r:
                eid, uid, cid = line.strip().split(',')
                # enrollments[eid] = (uid, cid)
                courses_by_user.setdefault(uid, []).append((eid, cid))

                if str.isdigit(eid):
                    if cid not in course_counts:
                        course_counts[cid] = 1.0
                    else:
                        course_counts[cid] += 1.0
                    if eid in labels:
                        l = labels[eid]
                        if l == 1:
                            if cid not in course_drops:
                                course_drops[cid] = 1.0
                            else:
                                course_drops[cid] += 1.0

        with open(test_enrollment_path, 'r') as r:
            for line in r:
                eid, uid, cid = line.strip().split(',')
                # enrollments[eid] = (uid, cid)
                courses_by_user.setdefault(uid, []).append((eid, cid))

                if str.isdigit(eid):
                    if cid not in course_counts:
                        course_counts[cid] = 1.0
                    else:
                        course_counts[cid] += 1.0

                    if eid in labels:
                        l = labels[eid]
                        if l == 1:
                            if cid not in course_drops:
                                course_drops[cid] = 1.0
                            else:
                                course_drops[cid] += 1.0

        # self.enrollments = enrollments
        self.courses_by_user = courses_by_user

        user_drops = {}
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
            user_drops[uid] = (drops + 0.7) / (len(items) + 0.0)

        self.user_drops_ratio = user_drops

        course_db = {}
        print 'course_size: %d' % len(course_drops)
        for cid in course_counts.keys():
            # drops = 0
            # if cid in course_drops:
            #     drops = course_drops[cid]
            course_db[cid] = {'audience': course_counts[cid], 'drops': course_drops[cid]}

        self.course_db = course_db

        self.module_db = load_modules(module_path)
        self.module_db.order_modules()

        self.time_modules = {}

        self._train_log_csv = open(train_log_path, 'r')
        self._test_log_csv = open(test_log_path, 'r')

        # FeatureExtractor.__init__(self, mode, data_type, log_csv_path, feature_path, debug_limit)
        self._filtered_iter = self._mode_filter(self._train_log_csv, self._test_log_csv, mode)

    def _mode_filter(self, iter, iter2, mode):
        for cnt, line in enumerate(iter):
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

    # def _fullgroupby(seq, key):
    #   """groups items by key; seq's ordering doesn't matter.  unlike itertools.groupby and unlike unix uniq, but like sql group by."""
    #   dec = [ (key(x),x) for x in seq ]
    #   dec.sort()
    #   return ( (g, [x for k,x in vals])  for g,vals  in itertools.groupby(dec, lambda (k,x): k))

    def build(self):
        tuple_iter = list(self._tuple_generator(self._filtered_iter))
        # grouped_iter = itertools.groupby(tuple_iter, lambda x: x[0])

        # grouped_iter = groupby(tuple_iter, lambda x: x[0])
        gp_dict = {}
        for k, value in tuple_iter:
            # k = key(value)
            gp_dict.setdefault(k, []).append(value)

        grouped_iter = [(k, v) for k, v in gp_dict.items()]
        # dec = [(x[0], x) for x in tuple_iter]
        # dec.sort()
        # grouped_iter = ((g, [x for k, x in vals]) for g, vals in itertools.groupby(dec, lambda (k, x): k))
        # bag_iter = self._bag_generator(grouped_iter)

        self.cal_info(grouped_iter)
        self._train_log_csv.close()
        self._test_log_csv.close()

    def cal_info(self, iter):
        for cid, logs in iter:
            # print logs[0]
            # print 'LEN: %d' % len(logs)
            requests = sorted([(log['time'].strftime('%Y%m%d'), log['time']) for log in logs], key=lambda x: x[0])
            course_startday, start = requests[0]
            course_endday, end = requests[-1]
            course_duration = (end - start).days
            self.course_db[cid]['course_startday'] = datetime.datetime.strptime(course_startday, '%Y%m%d')
            self.course_db[cid]['course_endday'] = datetime.datetime.strptime(course_endday, '%Y%m%d')
            self.course_db[cid]['course_duration'] = course_duration
            # print 'CID: %s - %d' % (cid, len(requests))
            counts = {}
            max_cnt = 0.0
            for d, t in requests:
                counts.setdefault(d, []).append(t)
            for d, v in counts.items():
                cnt = len(v) + 0.0
                if cnt > max_cnt:
                    max_cnt = cnt
                counts[d] = cnt
            self.course_db[cid]['day_visits'] = counts
            self.course_db[cid]['day_maxvisit'] = max_cnt

            scale_counts = {}
            for d, c in counts.items():
                scale_counts[d] = c / max_cnt
            self.course_db[cid]['scale_visits'] = scale_counts

            for log in logs:
                if log['event'] == 'access':
                    t = log['time']
                    o = log['object']
                    #if self.module_db.exist(o):
                    self.time_modules.setdefault(o, []).append(t)


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
            # print k
            # yield SimpleCourseFeatureBag(k, [t[1] for t in g], [], [])
            yield (k, g)
