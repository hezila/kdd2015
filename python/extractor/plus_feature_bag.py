#!/usr/bin/env python
#-*- coding: utf-8 -*-

from feature_bag import FeatureBag
import datetime
import time
import os
import math

import numpy as np

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__='02-07-2015'

event_types = ['problem', 'video', 'access', 'wiki', 'discussion', 'nagivate', 'page_close']


class PlusFeatureBag(FeatureBag):

    def __init__(self, enrollment_id, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)
        self.logs = sorted(logs, key = lambda x: x['time'])

    def extract_user_features(self, db):
        uid = self.logs[0]['user_name']
        cnt = 0.0
        if uid in db.courses_by_user:
            cnt = len(db.courses_by_user[uid]) + 0.0
        self.feature_keys.append('user_courses_count')
        self.feature_values.append(cnt)

        user_drops = 0.0
        user_drop_ratio = 0.8
        if uid in db.user_drops:
            user_drops = db.user_drops[uid] - 1.0
            user_drop_ratio = db.user_drops_ratio[uid]
        #self.feature_keys.append('user_drops_count')
        #self.feature_values.append(user_drops)
        self.feature_keys.append('user_drop_ratio')
        self.feature_values.append(user_drop_ratio)
        return self

    def extract_course_features(self, db):
        cid = db.course_by_eid[self.enrollment_id]
        all_audience = db.course_db[cid]['all_audience']
        pop = db.course_db[cid]['course_pop']
        drops = db.course_db[cid]['drops']
        drop_ratio = db.course_db[cid]['drop_ratio']
        self.feature_keys.append('audience')
        self.feature_values.append(all_audience)
        self.feature_keys.append('course_pop')
        self.feature_values.append(pop)
        self.feature_keys.append('drops')
        self.feature_values.append(drops)
        self.feature_keys.append('drop_ratio')
        self.feature_values.append(drop_ratio)


        return self

    def extract_visit_features(self, db):
        cid = db.course_by_eid[self.enrollment_id]
        visits = db.course_db[cid]['day_visits']
        scale_visits = db.course_db[cid]['scale_visits']

        log_start = self.logs[0]['time'].strftime('%Y%m%d')

        v = 0.0
        sv = 0.0
        if log_start in visits:
            v = visits[log_start]
            sv = scale_visits[log_start]
        self.feature_keys.append('fst_day_pop')
        self.feature_values.append(v)
        self.feature_keys.append('fst_day_scale_pop')
        self.feature_values.append(sv)

        log_end = self.logs[0]['time'].strftime('%Y%m%d')
        v = 0.0
        sv = 0.0
        if log_end in visits:
            v = visits[log_end]
            sv = scale_visits[log_end]
        self.feature_keys.append('end_day_pop')
        self.feature_values.append(v)
        self.feature_keys.append('end_day_scale_pop')
        self.feature_values.append(sv)

        pops = [log['time'].strftime('%Y%m%d') for log in self.logs]
        scale_pops = []
        for d in pops:
            p = 0.0
            if d in scale_visits:
                p = scale_visits[d]
            scale_pops.append(p)
        self.feature_keys.append('max_visit_pop')
        self.feature_values.append(max(scale_pops))
        self.feature_keys.append('min_visit_pop')
        self.feature_values.append(min(scale_pops))
        self.feature_keys.append('mean_visit_pop')
        self.feature_values.append(sum(scale_pops) / (len(scale_pops) + 0.0))
        return self

    def extract_moduletime_features(self, db):
        time_modules = db.time_modules
        time_videos = db.time_videos
        cid = db.course_by_eid[self.enrollment_id]

        modules = db.module_db.modules_by_cid[cid]
        chapters = db.module_db.chapters_by_cid[cid]
        sqs = db.module_db.sqs_by_cid[cid]
        videos = db.module_db.videos_by_cid[cid]

        module_access_times = {}
        chapter_access_times = {}
        sqs_access_times = {}
        videos_access_times = {}
        for log in self.logs:
            o = log['object']
            if o in modules:
                module_access_times.setdefault(o, []).append(log['time'])
            if o in chapters:
                chapter_access_times.setdefault(o, []).append(log['time'])
            if o in sqs:
                sqs_access_times.setdefault(o, []).append(log['time'])
            if o in videos:
                videos_access_times.setdefault(o, []).append(log['time'])

        for mid in module_access_times.keys():
            module_access_times[mid] = min(module_access_times[mid])
        for mid in chapter_access_times.keys():
            chapter_access_times[mid] = min(chapter_access_times[mid])
        for mid in sqs_access_times.keys():
            sqs_access_times[mid] = min(sqs_access_times[mid])
        for mid in videos_access_times.keys():
            videos_access_times[mid] = min(videos_access_times[mid])

        module_lags = []
        for mid in module_access_times.keys():
            if mid not in time_modules:
                continue
            s = min(time_modules[mid])
            t = module_access_times[mid]
            l = (t - s).days
            module_lags.append(l)

        chapter_lags = []
        for mid in chapter_access_times.keys():
            if mid not in time_modules:
                continue
            s = min(time_modules[mid])
            t = chapter_access_times[mid]
            l = (t - s).days
            chapter_lags.append(l)

        sqs_lags = []
        for mid in sqs_access_times.keys():
            if mid not in time_modules:
                continue
            s = min(time_modules[mid])
            t = sqs_access_times[mid]
            l = (t  - s).days
            sqs_lags.append(l)

        video_lags = []
        for mid in videos_access_times.keys():
            if mid not in time_videos:
                continue
            s = min(time_videos[mid])
            t = videos_access_times[mid]
            l = (t - s).days
            video_lags.append(l)


        if len(module_lags) == 0:
            module_lags.append(300)
        if len(chapter_lags) == 0:
            chapter_lags.append(300)
        if len(sqs_lags) == 0:
            sqs_lags.append(300)
        if len(video_lags) == 0:
            video_lags.append(300)

        self.feature_keys.append('max_module_lag')
        self.feature_values.append(max(module_lags))
        self.feature_keys.append('max_chapter_lag')
        self.feature_values.append(max(chapter_lags))
        self.feature_keys.append('max_sqs_lag')
        self.feature_values.append(max(sqs_lags))
        self.feature_keys.append('max_video_lag')
        self.feature_values.append(max(video_lags))

        self.feature_keys.append('min_module_lag')
        self.feature_values.append(min(module_lags))
        self.feature_keys.append('min_chapter_lag')
        self.feature_values.append(min(chapter_lags))
        self.feature_keys.append('min_sqs_lags')
        self.feature_values.append(min(sqs_lags))
        self.feature_keys.append('min_video_lag')
        self.feature_values.append(min(video_lags))

        self.feature_keys.append('sum_module_lag')
        self.feature_values.append(sum(module_lags))
        self.feature_keys.append('sum_chapter_lag')
        self.feature_values.append(sum(chapter_lags))
        self.feature_keys.append('sum_sqs_lag')
        self.feature_values.append(sum(sqs_lags))
        self.feature_keys.append('sum_video_lag')
        self.feature_values.append(sum(video_lags))

        self.feature_keys.append('mean_module_lag')
        self.feature_values.append(sum(module_lags) / (len(module_lags) + 0.0))
        self.feature_keys.append('mean_chapter_lag')
        self.feature_values.append(sum(chapter_lags) / (len(chapter_lags) + 0.0))
        self.feature_keys.append('mean_sqs_lag')
        self.feature_values.append(sum(sqs_lags) / (len(sqs_lags) + 0.0))
        self.feature_keys.append('mean_video_lag')
        self.feature_values.append(sum(video_lags) / (len(video_lags) + 0.0))

        return self

    def extract_module_features(self, db):
        cid = db.course_by_eid[self.enrollment_id]
        course_start, course_end = db.course_dates[cid]

        chapters = db.module_db.chapters_by_cid[cid]
        sqs = db.module_db.sqs_by_cid[cid]
        videos = db.module_db.videos_by_cid[cid]

        self.feature_keys.append('course_chapter_count')
        self.feature_values.append(len(chapters))
        self.feature_keys.append('course_sequences_count')
        self.feature_values.append(len(sqs))
        self.feature_keys.append('course_video_count')
        self.feature_values.append(len(videos))

        hit_chapters = [log['object'] for log in self.logs if log['object'] in chapters]
        hit_sqs = [log['object'] for log in self.logs if log['object'] in sqs]

        self.feature_keys.append('chapter_access_count')
        self.feature_values.append(len(hit_chapters))
        self.feature_keys.append('sqs_access_count')
        self.feature_values.append(len(hit_sqs))

        chapter_counts = {}
        for mid in hit_chapters:
            chapter_counts.setdefault(mid, 0.0)
            chapter_counts[mid] += 1.0
        counts = []
        for mid in chapter_counts.keys():
            counts.append(chapter_counts[mid])
        max_cnt = 0.0
        mean_cnt = 0.0
        if len(chapter_counts)  > 0:
            max_cnt = max(counts)
            mean_cnt = sum(counts) / (len(counts) + 0.0)
        self.feature_keys.append('max_chapter_access_count')
        self.feature_values.append(max_cnt)
        self.feature_keys.append('mean_chapter_access_count')
        self.feature_values.append(mean_cnt)

        sqs_counts = {}
        for mid in hit_sqs:
            sqs_counts.setdefault(mid, 0.0)
            sqs_counts[mid] += 1.0
        counts = []
        for mid in sqs_counts.keys():
            counts.append(sqs_counts[mid])
        max_cnt = 0.0
        mean_cnt = 0.0
        if len(sqs_counts)  > 0:
            max_cnt = max(counts)
            mean_cnt = sum(counts) / (len(counts) + 0.0)
        self.feature_keys.append('max_sqs_access_count')
        self.feature_values.append(max_cnt)
        self.feature_keys.append('mean_sqs_access_count')
        self.feature_values.append(mean_cnt)


        hit_chapters = list(set(hit_chapters))
        hit_sqs = list(set(hit_sqs))

        #hit_videos = list(set([log['object'] for log in self.logs in log['object'] in videos]))
        self.feature_keys.append('hit_chapter_count')
        self.feature_values.append(len(hit_chapters))
        self.feature_keys.append('left_chapter_count')
        self.feature_values.append(len(chapters) - len(hit_chapters))
        self.feature_keys.append('hit_chapter_ratio')
        self.feature_values.append(len(hit_chapters) / (len(chapters) + 0.0))

        self.feature_keys.append('hit_sqs_count')
        self.feature_values.append(len(hit_sqs))
        self.feature_keys.append('left_sqs_count')
        self.feature_values.append(len(sqs) - len(hit_sqs))
        self.feature_keys.append('hit_sqs_ratio')
        self.feature_values.append(len(hit_sqs) / (len(sqs) + 0.0))

        request_dates = [log['object'] for log in self.logs if log['event'] == 'access']
        request_dates = [db.module_db.modules[o].get_cate() for o in request_dates if o in  db.module_db.modules]

        module_counts = {}
        for c in request_dates:
            module_counts.setdefault(c, 0)
            module_counts[c] += 1.0
        for c in db.module_cates:
            cnt = 0
            if c in module_counts:
                cnt = module_counts[c]
            self.feature_keys.append('module_cate_%s_count' % c)
            self.feature_values.append(cnt)

        course_aft1week = course_start + datetime.timedelta(days=7)
        course_aft2week = course_start + datetime.timedelta(days=14)
        course_bf1week = course_end - datetime.timedelta(days=7)
        course_bf2week = course_end - datetime.timedelta(days=14)


        chapter_requests = [log['object'] for log in self.logs if log['object'] in chapters and log['time'] <= course_aft1week]
        self.feature_keys.append('chapter_access_count_aft1week')
        self.feature_values.append(len(chapter_requests))
        chapter_counts = {}
        for mid in chapter_requests:
            chapter_counts.setdefault(mid, 0.0)
            chapter_counts[mid] += 1.0
        self.feature_keys.append('chapter_count_aft1week')
        self.feature_values.append(len(chapter_counts))
        freq = 0.0
        if len(hit_chapters) > 0:
            freq = len(chapter_counts) / (len(hit_chapters) + 0.0)
        self.feature_keys.append('chapter_access_freq_aft1week')
        self.feature_values.append(freq)
        counts = []
        for mid in chapter_counts.keys():
            counts.append(chapter_counts[mid])

        max_cnt = 0.0
        if len(chapter_counts) > 0:
            max_cnt = max(counts)
        self.feature_keys.append('chapter_max_access_count_aft1week')
        self.feature_values.append(max_cnt)

        chapter_requests = [log['object'] for log in self.logs if log['object'] in chapters and log['time'] <= course_aft2week]
        self.feature_keys.append('chapter_count_aft2week')
        self.feature_values.append(len(chapter_requests))
        chapter_counts = {}
        for mid in chapter_requests:
            chapter_counts.setdefault(mid, 0.0)
            chapter_counts[mid] += 1.0
        self.feature_keys.append('chapter_count_aft2week')
        self.feature_values.append(len(chapter_counts))
        freq = 0.0
        if len(hit_chapters) > 0:
            freq = len(chapter_counts) / (len(hit_chapters) + 0.0)
        self.feature_keys.append('chapter_access_freq_aft2week')
        self.feature_values.append(freq)

        counts = []
        for mid in chapter_counts.keys():
            counts.append(chapter_counts[mid])
        max_cnt = 0.0
        if len(chapter_counts) > 0:
            max_cnt = max(counts)
        self.feature_keys.append('chapter_max_access_count_aft2week')
        self.feature_values.append(max_cnt)



        chapter_requests = [log['object'] for log in self.logs if log['object'] in chapters and log['time'] >= course_bf1week]
        self.feature_keys.append('chapter_count_bf1week')
        self.feature_values.append(len(chapter_requests))

        chapter_counts = {}
        for mid in chapter_requests:
            chapter_counts.setdefault(mid, 0.0)
            chapter_counts[mid] += 1.0
        self.feature_keys.append('chapter_count_bf1week')
        self.feature_values.append(len(chapter_counts))
        freq = 0.0
        if len(hit_chapters) > 0:
            freq = len(chapter_counts) / (len(hit_chapters) + 0.0)
        self.feature_keys.append('chapter_access_freq_bf1week')
        self.feature_values.append(freq)

        counts = []
        for mid in chapter_counts.keys():
            counts.append(chapter_counts[mid])
        max_cnt = 0.0
        if len(chapter_counts) > 0:
            max_cnt = max(counts)
        self.feature_keys.append('chapter_max_access_count_bf1week')
        self.feature_values.append(max_cnt)



        chapter_requests = [log['object'] for log in self.logs if log['object'] in chapters and log['time'] >= course_bf2week]
        self.feature_keys.append('chapter_count_bf2week')
        self.feature_values.append(len(chapter_requests))
        chapter_counts = {}
        for mid in chapter_requests:
            chapter_counts.setdefault(mid, 0.0)
            chapter_counts[mid] += 1.0
        self.feature_keys.append('chapter_count_bf2week')
        self.feature_values.append(len(chapter_counts))
        freq = 0.0
        if len(hit_chapters) > 0:
            freq = len(chapter_counts) / (len(hit_chapters) + 0.0)
        self.feature_keys.append('chapter_access_freq_bf2week')
        self.feature_values.append(freq)
        counts = []
        for mid in chapter_counts.keys():
            counts.append(chapter_counts[mid])
        max_cnt = 0.0
        if len(chapter_counts) > 0:
            max_cnt = max(counts)
        self.feature_keys.append('chapter_max_access_count_bf2week')
        self.feature_values.append(max_cnt)



        sqs_requests = [log['object'] for log in self.logs if log['object'] in sqs and log['time'] >= course_bf1week]
        self.feature_keys.append('sqs_count_bf1week')
        self.feature_values.append(len(sqs_requests))

        sqs_counts = {}
        for mid in sqs_requests:
            sqs_counts.setdefault(mid, 0.0)
            sqs_counts[mid] += 1.0
        self.feature_keys.append('sqs_count_bf1week')
        self.feature_values.append(len(sqs_counts))
        freq = 0.0
        if len(hit_sqs) > 0:
            freq = len(sqs_counts) / (len(hit_sqs) + 0.0)
        self.feature_keys.append('sqs_access_freq_bf1week')
        self.feature_values.append(freq)

        counts = []
        for mid in sqs_counts.keys():
            counts.append(sqs_counts[mid])
        max_cnt = 0.0
        if len(sqs_counts) > 0:
            max_cnt = max(counts)
        self.feature_keys.append('sqs_max_access_count_bf1week')
        self.feature_values.append(max_cnt)



        sqs_requests = [log['object'] for log in self.logs if log['object'] in sqs and log['time'] >= course_bf2week]
        self.feature_keys.append('sqs_count_bf2week')
        self.feature_values.append(len(sqs_requests))
        sqs_counts = {}
        for mid in sqs_requests:
            sqs_counts.setdefault(mid, 0.0)
            sqs_counts[mid] += 1.0
        self.feature_keys.append('sqs_count_bf2week')
        self.feature_values.append(len(sqs_counts))
        freq = 0.0
        if len(hit_sqs) > 0:
            freq = len(sqs_counts) / (len(hit_sqs) + 0.0)
        self.feature_keys.append('sqs_access_freq_bf2week')
        self.feature_values.append(freq)
        counts = []
        for mid in sqs_counts.keys():
            counts.append(sqs_counts[mid])
        max_cnt = 0.0
        if len(sqs_counts) > 0:
            max_cnt = max(counts)
        self.feature_keys.append('sqs_max_access_count_bf2week')
        self.feature_values.append(max_cnt)



        sqs_requests = [log['object'] for log in self.logs if log['object'] in sqs and log['time'] <= course_aft1week]
        self.feature_keys.append('sqs_count_aft1week')
        self.feature_values.append(len(sqs_requests))
        sqs_counts = {}
        for mid in sqs_requests:
            sqs_counts.setdefault(mid, 0.0)
            sqs_counts[mid] += 1.0
        self.feature_keys.append('sqs_count_aft1week')
        self.feature_values.append(len(sqs_counts))
        freq = 0.0
        if len(hit_sqs) > 0:
            freq = len(sqs_counts) / (len(hit_sqs) + 0.0)
        self.feature_keys.append('sqs_access_freq_aft1week')
        self.feature_values.append(freq)
        counts = []
        for mid in sqs_counts.keys():
            counts.append(sqs_counts[mid])
        max_cnt = 0.0
        if len(sqs_counts) > 0:
            max_cnt = max(counts)
        self.feature_keys.append('sqs_max_access_count_aft1week')
        self.feature_values.append(max_cnt)



        sqs_requests = [log['object'] for log in self.logs if log['object'] in sqs and log['time'] <= course_aft2week]
        self.feature_keys.append('sqs_count_aft2week')
        self.feature_values.append(len(sqs_requests))
        sqs_counts = {}
        for mid in sqs_requests:
            sqs_counts.setdefault(mid, 0.0)
            sqs_counts[mid] += 1.0
        self.feature_keys.append('sqs_count_aft2week')
        self.feature_values.append(len(sqs_counts))
        freq = 0.0
        if len(hit_sqs) > 0:
            freq = len(sqs_counts) / (len(hit_sqs) + 0.0)
        self.feature_keys.append('sqs_access_freq_aft2week')
        self.feature_values.append(freq)
        counts = []
        for mid in sqs_counts.keys():
            counts.append(sqs_counts[mid])
        max_cnt = 0.0
        if len(sqs_counts) > 0:
            max_cnt = max(counts)
        self.feature_keys.append('sqs_max_access_count_aft2week')
        self.feature_values.append(max_cnt)




        cate_counts = {}
        request_dates = [log['object'] for log in self.logs if log['object'] in db.module_db.modules and log['time'] <= course_aft1week]
        request_dates = [db.module_db.modules[mid].get_cate() for mid in request_dates]
        for c in request_dates:
            cate_counts.setdefault(c, 0)
            cate_counts[c] += 1.0

        for c in db.module_cates:
            cnt = 0
            if c in cate_counts:
                cnt = cate_counts[c]
            self.feature_keys.append('module_cate_{0:s}_count_aft1week'.format(c))
            self.feature_values.append(cnt)

        cate_counts = {}
        request_dates = [log['object'] for log in self.logs if log['object'] in db.module_db.modules and log['time'] <= course_aft2week]
        request_dates = [db.module_db.modules[mid].get_cate() for mid in request_dates]
        for c in request_dates:
            cate_counts.setdefault(c, 0.0)
            cate_counts[c] += 1.0
        for c in db.module_cates:
            cnt = 0
            if c in request_dates:
                cnt = cate_counts[c]
            self.feature_keys.append('module_cate_{0:s}_count_aft2week'.format(c))
            self.feature_values.append(cnt)

        cate_counts = {}
        request_dates = [log['object'] for log in self.logs if log['object'] in db.module_db.modules and log['time'] >= course_bf1week]
        request_dates = [db.module_db.modules[mid].get_cate() for mid in request_dates]
        for c in request_dates:
            cate_counts.setdefault(c, 0)
            cate_counts[c] += 1.0

        for c in db.module_cates:
            cnt = 0.0
            if c in cate_counts:
                cnt = cate_counts[c]
            self.feature_keys.append('module_cate_{0:s}_count_bf1week'.format(c))
            self.feature_values.append(cnt)

        cate_counts = {}
        request_dates = [log['object'] for log in self.logs if log['object'] in db.module_db.modules and log['time'] >= course_bf2week]
        request_dates = [db.module_db.modules[mid].get_cate() for mid in request_dates]
        for c in request_dates:
            cate_counts.setdefault(c, 0)
            cate_counts[c] += 1.0
        for c in db.module_cates:
            cnt = 0.0
            if c in cate_counts:
                cnt = cate_counts[c]
            self.feature_keys.append('module_cate_{0:s}_count_bf2week'.format(c))
            self.feature_values.append(cnt)


        return self

    def extract_coursetime_features(self, db):
        cid = db.course_by_eid[self.enrollment_id]
        course_start, course_end = db.course_dates[cid]

        course_aft1week = course_start + datetime.timedelta(days=7)
        course_aft2week = course_start + datetime.timedelta(days=14)

        course_bf1week = course_end - datetime.timedelta(days=7)
        course_bf2week = course_end - datetime.timedelta(days=14)

        request_dates = sorted([(log['time'], log['event']) for log in self.logs], key=lambda x: x[0])
        start = request_dates[0][0]
        end = request_dates[-1][0]
        d = (end - start).days
        if d == 0:
            d = 1.0

        start_in1week = 0
        if start <= course_aft1week:
            start_in1week = 1
        start_in2week = 0
        if start <= course_aft2week:
            start_in2week = 1

        after_days = (start - course_start).days
        before_days = (course_end - end).days
        self.feature_keys.append('after_course_start_days')
        self.feature_values.append(after_days)
        self.feature_keys.append('after_course_start_days_ratio')
        self.feature_values.append(after_days / d)

        self.feature_keys.append('before_course_end_days')
        self.feature_values.append(before_days)
        self.feature_keys.append('before_course_end_days_ratio')
        self.feature_values.append(before_days / d)


        self.feature_keys.append('request_start_in1week')
        self.feature_values.append(start_in1week)

        self.feature_keys.append('request_start_in2week')
        self.feature_values.append(start_in2week)

        end_in1week = 0
        if end >= course_bf1week:
            end_in1week = 1
        end_in2week = 0
        if end >= course_bf2week:
            end_in2week = 1

        start_in1week = 0
        if start >= course_bf1week:
            start_in1week = 1
        start_in2week = 0
        if start >= course_bf2week:
            start_in2week = 1

        self.feature_keys.append('request_end_in1week')
        self.feature_values.append(end_in1week)
        self.feature_keys.append('request_end_in2week')
        self.feature_values.append(end_in2week)

        self.feature_keys.append('request_start_bf1week')
        self.feature_values.append(start_in1week)
        self.feature_keys.append('request_start_bf2week')
        self.feature_values.append(start_in2week)

        request_dates_in1week = [(log['time'].strftime('%Y%m%d'), log['event']) for log in self.logs if log['time'] <= course_aft1week]
        event_count = len(request_dates_in1week)
        self.feature_keys.append('event_count_in1week')
        self.feature_values.append(event_count)

        request_dates_in2week = [(log['time'].strftime('%Y%m%d'), log['event']) for log in self.logs if log['time'] <= course_aft2week]
        event_count = len(request_dates_in2week)
        self.feature_keys.append('event_count_in2week')
        self.feature_values.append(event_count)

        days = set([d for d, e in request_dates_in1week])
        self.feature_keys.append('active_days_in1week')
        self.feature_values.append(len(days))

        days = set([d for d, e in request_dates_in2week])
        self.feature_keys.append('active_days_in2week')
        self.feature_values.append(len(days))

        event_counts = {}
        for d, e in request_dates_in1week:
            event_counts.setdefault(e, 0)
            event_counts[e] += 1.0
        for e in event_types:
            cnt = 0.0
            if e in event_counts:
                cnt = event_counts[e]
            self.feature_keys.append('{0:s}_count_in1week'.format(e))
            self.feature_values.append(cnt)

        event_counts = {}
        for d, e in request_dates_in2week:
            event_counts.setdefault(e, 0)
            event_counts[e] += 1.0
        for e in event_types:
            cnt = 0.0
            if e in event_counts:
                cnt = event_counts[e]
            self.feature_keys.append('{0:s}_count_in2week'.format(e))
            self.feature_values.append(cnt)


        request_dates_bf1week = [(log['time'].strftime('%Y%m%d'), log['event']) for log in self.logs if log['time'] >= course_bf1week]

        request_dates_bf2week = [(log['time'].strftime('%Y%m%d'), log['event']) for log in self.logs if log['time'] >= course_bf2week]

        event_count = len(request_dates_bf1week)
        self.feature_keys.append('event_count_bf1week')
        self.feature_values.append(event_count)

        event_count = len(request_dates_bf2week)
        self.feature_keys.append('event_count_bf2week')
        self.feature_values.append(event_count)

        days = set([d for d, e in request_dates_bf1week])
        self.feature_keys.append('active_days_bf1week')
        self.feature_values.append(len(days))

        days = set([d for d, e in request_dates_bf2week])
        self.feature_keys.append('active_days_bf2week')
        self.feature_values.append(len(days))

        event_counts = {}
        for d, e in request_dates_bf1week:
            event_counts.setdefault(e, 0)
            event_counts[e] += 1.0

        for e in event_types:
            cnt = 0
            if e in event_counts:
                cnt = event_counts[e]
            self.feature_keys.append('{0:s}_count_bf1week'.format(e))
            self.feature_values.append(cnt)

        event_counts = {}
        for d, e in request_dates_bf2week:
            event_counts.setdefault(e, 0)
            event_counts[e] += 1.0
        for e in event_types:
            cnt = 0
            if e in event_counts:
                cnt = event_counts[e]
            self.feature_keys.append('{0:s}_count_bf2week'.format(e))
            self.feature_values.append(cnt)

        return self

    def extract_azure_feature(self):
        request_dates = sorted([(log['time'], int(log['time'].strftime('%U')), log['event']) for log in self.logs], key = lambda x: x[0])
        min_week = request_dates[0][1]
        max_week = request_dates[-1][1]

        min_year = int(request_dates[0][0].strftime('%Y'))
        max_year = int(request_dates[-1][0].strftime('%Y'))



        if max_year > min_year:
            week_last_day = datetime.date(min_year, 12, 31).isocalendar()[1]
            if week_last_day == 1:
                week_last_day = datetime.date(min_year, 12, 24).isocalendar()[1]

            for i in range(len(request_dates)):
                t, w, e = request_dates[i]
                y = int(t.strftime('%Y'))
                if y == max_year:
                    w += week_last_day
                    request_dates[i] = (t, w, e)

                if w == 1 and y == min_year:
                    w += week_last_day
                    request_dates[i] = (t, w, e)


        weeks = [w for t, w, e in request_dates]
        max_week = max(weeks)
        num_weeks = max_week - min_week + 1

        self.feature_keys.append('num_weeks')
        self.feature_values.append(num_weeks)

        event_counts = [0] * num_weeks
        weeks = range(num_weeks)
        for t, w, e in request_dates:
            w = w - min_week
            event_counts[w] += 1.0

        self.feature_keys.append('max_weekly_event_count')
        self.feature_values.append(max(event_counts))
        self.feature_keys.append('min_weekly_event_count')
        self.feature_values.append(min(event_counts))
        mean = sum(event_counts) / (num_weeks + 0.0)
        self.feature_keys.append('mean_weekly_event_count')
        self.feature_values.append(mean)
        std = 0.0
        if num_weeks > 1:
            for cnt in event_counts:
                std += (cnt - mean)**2
            std = math.sqrt(std / (num_weeks + 0.0))
        self.feature_keys.append('std_weekly_event_count')
        self.feature_values.append(std)

        event_count = np.array(event_counts)

        m = 0
        c = 0
        if num_weeks > 1:
            A = np.vstack([weeks, np.ones(len(weeks))]).T
            m, c = np.linalg.lstsq(A, event_counts)[0]

        self.feature_keys.append('week_trend_m')
        self.feature_values.append(m)
        self.feature_keys.append('week_trend_c')
        self.feature_values.append(c)


        z1 = 0
        z2 = 0
        if num_weeks > 2:
            #fit a polynomial model y = a + bx + cx**2
            z = np.polyfit(weeks, event_counts, 2)
            z1 = z[1]
            z2 = z[2]

        self.feature_keys.append('poly_fit_z1')
        self.feature_values.append(z1)
        self.feature_keys.append('poly_fit_z2')
        self.feature_values.append(z2)

        m = 0
        c = 0
        if num_weeks > 1:
            pdf = 0.0
            for i in range(len(event_counts)):
                cnt = event_counts[i]
                event_counts[i] += pdf
                pdf += cnt
                m, c = np.linalg.lstsq(A, event_counts)[0]
        self.feature_keys.append('week_pdf_week_m')
        self.feature_values.append(m)
        self.feature_keys.append('week_pdf_week_c')
        self.feature_values.append(c)


        return self
