#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import Counter
import datetime
import time
import os
import math

from feature_bag import FeatureBag
from utils import *

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-18-2015'

base_dir = os.path.dirname(__file__)
event_types = ['problem', 'video', 'access', 'wiki', 'discussion', 'nagivate', 'page_close']
server_events = ['access', 'wiki', 'discussion', 'problem', 'nagivate']
browser_events = ['access', 'video', 'problem', 'page_close']

class SimpleFeatureBag(FeatureBag):
    def __init__(self, enrollment_id, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)
        self.logs = sorted(logs, key=lambda x: x['time'])
        self.start = self.logs[0]['time'].strftime('%Y%m%d')
        self.start_day = datetime.datetime.strptime(self.start, '%Y%m%d')

        self.end = self.logs[-1]['time'].strftime('%Y%m%d')
        self.end_day = datetime.datetime.strptime(self.end, '%Y%m%d')

    def extract_request_year(self):
        y = int(self.start_day.strftime('%Y%m'))
        if y <= 201402:
            self.feature_keys.append('year_2013')
            self.feature_values.append(1)
            self.feature_keys.append('year_2014')
            self.feature_values.append(0)
        else:
            self.feature_keys.append('year_2013')
            self.feature_values.append(0)
            self.feature_keys.append('year_2014')
            self.feature_values.append(1)
        return self

    def extract_request_mean_time(self):
        sum_time = reduce(lambda x, y: x+y, [time.mktime(log['time'].timetuple()) for log in self.logs])
        mean_time = sum_time / (len(self.logs) + 0.0)
        mean_time = datetime.datetime.fromtimestamp(mean_time)
        m = int(mean_time.strftime('%m'))
        d = int(mean_time.strftime('%d'))
        # print 'm: %d - d: %d' % (m, d)
        for i in range(1, 13):
            self.feature_keys.append('month_%d' % i)
            v = 0
            if m == i:
                v = 1
            self.feature_values.append(v)
        for i in range(1, 31):
            self.feature_keys.append('day_%d'% i)
            v = 0
            if m == i:
                v = 1
            elif m >= 30 and i == 30:
                v = 1
            self.feature_values.append(v)
        return self

    def extract_request_exam_holiday(self):
        fst_m = int(self.start_day.strftime('%m'))

        if fst_m in [7, 1]:
            self.feature_keys.append('exam_time_fstday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('exam_time_fstday')
            self.feature_values.append(0)

        if fst_m in [7, 8, 2]:
            self.feature_keys.append('holiday_fstday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('holiday_fstday')
            self.feature_values.append(0)

        lst_m = int(self.end_day.strftime('%m'))
        if lst_m in [7, 1]:
            self.feature_keys.append('exam_time_lstday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('exam_time_lstday')
            self.feature_values.append(0)

        if lst_m in [7, 8, 2]:
            self.feature_keys.append('holiday_lstday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('holiday_lstday')
            self.feature_values.append(0)

        if fst_m in [7, 1] and lst_m in [7, 1]:
            self.feature_keys.append('in_exam')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('in_exam')
            self.feature_values.append(0)

        if fst_m in [7, 8, 2] and lst_m in [7, 8, 2]:
            self.feature_keys.append('in_holiday')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('in_holiday')
            self.feature_values.append(0)

        return self

    def extract_duration_days(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        duration = (request_dates[-1] - request_dates[0]).days
        if duration == 0:
            duration = 1.0

        self.feature_keys.append('duration')
        self.feature_values.append(duration + 0.0)


        if duration <= 2:
            self.feature_keys.append('duration_less2day')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('duration_less2day')
            self.feature_values.append(0)

        if duration <= 7 and duration > 2:
            self.feature_keys.append('duration_less1week')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('duration_less1week')
            self.feature_values.append(0)

        if duration > 7 and duration <= 14:
            self.feature_keys.append('duration_2week')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('duration_2week')
            self.feature_values.append(0)

        if duration >= 15:
            self.feature_keys.append('duration_over2weeks')
            self.feature_values.append(1)
        else:
            self.feature_keys.append('duration_over2weeks')
            self.feature_values.append(0)
        return self

    def extract_request_count(self):
        # ignore the page close event
        requests = [log for log in self.logs if log['event'] != 'page_close']
        x = len(requests) + 0.0
        self.feature_keys.append("request_count")
        self.feature_values.append(x)
        return self

    def extract_request_count_lst2week(self):
        # ignore the page close event
        start = self.end_day - datetime.timedelta(days=14)
        requests = [log for log in self.logs if (log['event'] != 'page_close' and log['time'] >= start)]
        x = len(requests) + 0.0
        self.feature_keys.append("request_count_lst2week")
        self.feature_values.append(x)
        return self

    def extract_request_count_lst1week(self):
        # ignore the page close event
        start = self.end_day - datetime.timedelta(days=7)
        requests = [log for log in self.logs if (log['event'] != 'page_close' and log['time'] >= start)]
        x = len(requests) + 0.0
        self.feature_keys.append("request_count_lst1week")
        self.feature_values.append(x)
        return self


    def extract_event_count(self):
        x = len(self.logs) + 0.0
        self.feature_keys.append("event_count")
        self.feature_values.append(x)
        return self

    def extract_event_count_lst2week(self):
        start = self.end_day - datetime.timedelta(days=14)
        requests = [log for log in self.logs if log['time'] >= start]
        x = len(requests) + 0.0
        self.feature_keys.append("event_count_lst2week")
        self.feature_values.append(x)
        return self

    def extract_event_count_lst1week(self):
        start = self.end_day - datetime.timedelta(days=7)
        requests = [log for log in self.logs if log['time'] >= start]
        x = len(requests) + 0.0
        self.feature_keys.append("event_count_lst1week")
        self.feature_values.append(x)
        return self

    def extract_active_days(self):
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs])
        days = len(request_dates) + 0.0

        self.feature_keys.append('active_days')
        self.feature_values.append(days)

        request_dates = sorted(list(request_dates))
        x = []
        w = self.start_day.weekday()
        if w >= 4:
            end = self.start_day + datetime.timedelta(days=15-w)
        else:
            end = self.start_day + datetime.timedelta(days=8-w)
        active_days = []
        i = 0
        while i < len(request_dates):
            d = request_dates[i]
            d = datetime.datetime.strptime(d, '%Y%m%d')
            if d <= end:
                active_days.append(d)
            else:
                x.append(len(active_days))
                active_days = [d]
                end = end + datetime.timedelta(days=7)
            i += 1
        if len(active_days) > 0:
            x.append(len(active_days))

        self.feature_keys.append('max_days_per_week')
        self.feature_values.append(max(x))

        self.feature_keys.append('min_days_per_week')
        self.feature_values.append(min(x))

        self.feature_keys.append('mean_days_per_week')
        self.feature_values.append(sum(x) / (len(x) + 0.0))


        duration = (self.end_day - self.start_day).days
        if duration == 0:
            duration = 1.0
        x = days / duration
        self.feature_keys.append('active_days_freq')
        self.feature_values.append(x)
        return self

    def extract_active_days_per_week(self):
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs])
        d = len(request_dates) + 0.0

        w = (self.end_day - self.start_day).days / 7.0
        if w == 0:
            w = 1.0
        x = d / w
        self.feature_keys.append('active_days_per_week')
        self.feature_values.append(x)
        return self

    def extract_active_days_lst2week(self):
        start = self.end_day - datetime.timedelta(days=14)
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs if log['time'] >= start])
        days = len(request_dates) + 0.0

        self.feature_keys.append('active_days_lst2week')
        self.feature_values.append(days)

        request_dates = sorted(list(request_dates))[::-1]
        w = self.end_day.weekday()
        end = self.end_day - datetime.timedelta(days=13+w)
        cut = end + datetime.timedelta(days=7)
        i = 0
        x1 = 0
        x2 = 0
        while i < len(request_dates):
            d = request_dates[i]
            d = datetime.datetime.strptime(d, '%Y%m%d')
            if d < end:
                break
            if d >= cut:
                x2 += 1.0
            else:
                x1 += 1.0
            i += 1

        self.feature_keys.append('max_days_lst2week')
        self.feature_values.append(max([x1, x2]))

        self.feature_keys.append('min_days_lst2week')
        self.feature_values.append(min([x1, x2]))

        self.feature_keys.append('mean_days_lst2week')
        self.feature_values.append((x1 + x2) / 2.0)

        return self

    def extract_active_days_lst1week(self):
        start = self.end_day - datetime.timedelta(days=7)
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs if log['time'] >= start])
        days = len(request_dates) + 0.0

        self.feature_keys.append('active_days_lst1week')
        self.feature_values.append(days)
        return self


    def extract_server_events(self):
        events = [log['event'] for log in self.logs if log['source'] == 'server']
        counts = {}
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1
        for e in server_events:
            cnt = 0.0
            if e in counts:
                cnt = counts[e]
            self.feature_keys.append('server_{0}_count'.format(e))
            self.feature_values.append(cnt)
        return self

    def extract_server_events_lst2week(self):
        start = self.end_day - datetime.timedelta(days=14)
        events = [log['event'] for log in self.logs if (log['source'] == 'server' and log['time'] >= start)]
        counts = {}
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1
        for e in server_events:
            cnt = 0.0
            if e in counts:
                cnt = counts[e]
            self.feature_keys.append('server_{0}_count_lst2week'.format(e))
            self.feature_values.append(cnt)
        return self

    def extract_server_events_lst1week(self):
        start = self.end_day - datetime.timedelta(days=7)
        events = [log['event'] for log in self.logs if (log['source'] == 'server' and log['time'] >= start)]
        counts = {}
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1
        for e in server_events:
            cnt = 0.0
            if e in counts:
                cnt = counts[e]
            self.feature_keys.append('server_{0}_count_lst1week'.format(e))
            self.feature_values.append(cnt)
        return self

    def extract_access_count(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if log['event'] == 'access']
        self.feature_keys.append('access_count')
        self.feature_values.append(len(requests) + 0.0)

        daily_access = {}
        for d, t in requests:
            daily_access.setdefault(d, []).append(t)

        modules = []
        for d, vs in daily_access.items():
            for v in vs:
                if v not in modules:
                    modules.append(v)
        self.feature_keys.append('module_count')
        self.feature_values.append(len(modules) + 0.0)

        modules = []
        for d, vs in daily_access.items():
            modules.append(len(set(vs)))

        max_m = 0
        median_m = 0
        if len(modules) > 0:
            max_m = max(modules)
            median_m = median(modules)
        self.feature_keys.append('max_module_per_day')
        self.feature_values.append(max_m)
        #self.feature_keys.append('min_module_per_day')
        #self.feature_values.append(min(modules))
        self.feature_keys.append('median_module_per_day')
        self.feature_values.append(median_m)

        avg = 0.0
        if len(modules) > 0:
            avg = sum(modules) / (len(modules) + 0.0)
        self.feature_keys.append('mean_module_per_day')
        self.feature_values.append(avg)

        std = 0.0
        for v in modules:
            std += (v - avg)**2
        if len(modules) > 0:
            std = math.sqrt(std / len(modules) + 0.0)
        self.feature_keys.append('std_module_per_day')
        self.feature_values.append(std)

        return self

    def extract_access_count_lstweek(self, w):
        start = self.end_day - datetime.timedelta(days=7 * w)
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if (log['event'] == 'access' and log['time'] >= start)]
        self.feature_keys.append('access_count_lst%dweek' % w)
        self.feature_values.append(len(requests) + 0.0)

        daily_access = {}
        for d, t in requests:
            daily_access.setdefault(d, []).append(t)

        modules = []
        for d, vs in daily_access.items():
            for v in vs:
                if v not in modules:
                    modules.append(v)
        self.feature_keys.append('module_count_lst%dweek' % w)
        self.feature_values.append(len(modules) + 0.0)

        modules = []
        for d, vs in daily_access.items():
            modules.append(len(set(vs)))

        max_m = 0
        median_m = 0
        if len(modules) > 0:
            max_m = max(modules)
            median_m = median(modules)

        self.feature_keys.append('max_module_lst%dweek' % w)
        self.feature_values.append(max_m)

        #self.feature_keys.append('min_module_lst2week')
        #self.feature_values.append(min(modules))

        self.feature_keys.append('median_module_lst%dweek' % w)
        self.feature_values.append(median_m)

        avg = 0.0
        if len(modules) > 0:
            avg = sum(modules) / (len(modules) + 0.0)
        self.feature_keys.append('mean_module_lst%dweek' % w)
        self.feature_values.append(avg)



        return self


    def extract_video_count(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if log['event'] == 'video']
        self.feature_keys.append('video_request_count')
        self.feature_values.append(len(requests) + 0.0)

        daily_videos = {}
        for d, t in requests:
            daily_videos.setdefault(d, []).append(t)

        videos = []
        for d, vs in daily_videos.items():
            for v in vs:
                if v not in videos:
                    videos.append(v)
        self.feature_keys.append('video_count')
        self.feature_values.append(len(videos) + 0.0)

        videos = []
        for d, vs in daily_videos.items():
            videos.append(len(set(vs)))

        max_v = 0
        median_v = 0
        if len(videos) > 0:
            max_v = max(videos)
            median_v = median(videos)

        self.feature_keys.append('max_video_per_day')
        self.feature_values.append(max_v)
        self.feature_keys.append('median_video_per_day')
        self.feature_values.append(median_v)


        avg = 0.0
        if len(videos) > 0:
            avg = sum(videos) / (len(videos) + 0.0)
        self.feature_keys.append('mean_video_per_day')
        self.feature_values.append(avg)

        return self


    def extract_video_count_lstweek(self, w):
        start = self.end_day - datetime.timedelta(days=7 * w)
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if log['event'] == 'video' and log['time'] >= start]
        self.feature_keys.append('video_request_count_lst%dweek' % w)
        self.feature_values.append(len(requests) + 0.0)

        daily_videos = {}
        for d, t in requests:
            daily_videos.setdefault(d, []).append(t)

        videos = []
        for d, vs in daily_videos.items():
            for v in vs:
                if v not in videos:
                    videos.append(v)
        self.feature_keys.append('video_count_lst%dweek' % w)
        self.feature_values.append(len(videos) + 0.0)

        videos = []
        for d, vs in daily_videos.items():
            videos.append(len(set(vs)))

        max_v = 0
        median_v = 0
        if len(videos) > 0:
            max_v = max(videos)
            median_v = median(videos)

        self.feature_keys.append('max_videos_perday_lst%dweek' % w)
        self.feature_values.append(max_v)
        self.feature_keys.append('median_videos_perday_lst%dweek' % w)
        self.feature_values.append(median_v)


        avg = 0.0
        if len(videos) > 0:
            avg = sum(videos) / (len(videos) + 0.0)
        self.feature_keys.append('mean_video_perday_lst%dweek' % w)
        self.feature_values.append(avg)

        return self


    def extract_problem_count(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if log['event'] == 'problem']
        self.feature_keys.append('problem_request_count')
        self.feature_values.append(len(requests) + 0.0)

        daily_problems = {}
        for d, t in requests:
            daily_problems.setdefault(d, []).append(t)

        problems = []
        for d, vs in daily_problems.items():
            for v in vs:
                if v not in problems:
                    problems.append(v)

        self.feature_keys.append('problem_count')
        self.feature_values.append(len(problems) + 0.0)

        problems = []
        for d, vs in daily_problems.items():
            problems.append(len(set(vs)))
        max_p = 0
        median_p = 0
        if len(problems) > 0:
            max_p = max(problems)
            median_p = median(problems)

        self.feature_keys.append('max_problem_per_day')
        self.feature_values.append(max_p)

        self.feature_keys.append('median_problem_per_day')
        self.feature_values.append(median_p)

        avg = 0.0
        if len(problems) > 0:
            avg = sum(problems) / (len(problems) + 0.0)
        self.feature_keys.append('mean_problem_per_day')
        self.feature_values.append(avg)
        return self

    def extract_problem_count_lst2week(self):
        start = self.end_day - datetime.timedelta(days=14)
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if (log['event'] == 'problem' and log['time'] >= start)]
        self.feature_keys.append('problem_request_count_lst2week')
        self.feature_values.append(len(requests) + 0.0)

        daily_problems = {}
        for d, t in requests:
            daily_problems.setdefault(d, []).append(t)

        problems = []
        for d, vs in daily_problems.items():
            for v in vs:
                if v not in problems:
                    problems.append(v)

        self.feature_keys.append('problem_count_lst2week')
        self.feature_values.append(len(problems) + 0.0)

        problems = []
        for d, vs in daily_problems.items():
            problems.append(len(set(vs)))
        max_p = 0
        median_p = 0
        if len(problems) > 0:
            max_p = max(problems)
            median_p = median(problems)
        self.feature_keys.append('max_problem_per_day_lst2week')
        self.feature_values.append(max_p)
        self.feature_keys.append('median_problem_per_day_lst2week')
        self.feature_values.append(median_p)

        avg = 0.0
        if len(problems) > 0:
            avg = sum(problems) / (len(problems) + 0.0)
        self.feature_keys.append('problem_per_day_lst2week')
        self.feature_values.append(avg)
        return self


    def extract_browser_events(self):
        events = [log['event'] for log in self.logs if log['source'] == 'browser']
        counts = {}
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1
        for e in browser_events:
            cnt = 0.0
            if e in counts:
                cnt = counts[e]
            self.feature_keys.append('browser_{0}_count'.format(e))
            self.feature_values.append(cnt)
        return self

    def extract_browser_events_lst2week(self):
        start = self.end_day - datetime.timedelta(days=14)

        events = [log['event'] for log in self.logs if (log['source'] == 'browser' and log['time'] >= start)]
        counts = {}
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1
        for e in browser_events:
            cnt = 0.0
            if e in counts:
                cnt = counts[e]
            self.feature_keys.append('browser_{0}_count_lst2week'.format(e))
            self.feature_values.append(cnt)
        return self

    def extract_browser_events_lst1week(self):
        start = self.end_day - datetime.timedelta(days=7)

        events = [log['event'] for log in self.logs if (log['source'] == 'browser' and log['time'] >= start)]
        counts = {}
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1
        for e in browser_events:
            cnt = 0.0
            if e in counts:
                cnt = counts[e]
            self.feature_keys.append('browser_{0}_count_lst1week'.format(e))
            self.feature_values.append(cnt)
        return self

    def extract_source_count(self):
        events = [log['source'] for log in self.logs]
        counts = {}
        counts['server'] = 0.0
        counts['browser'] = 0.0
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1.0


        for s in ['server', 'browser']:
            cnt = 0.0
            if s in counts:
                cnt = counts[s]
            self.feature_keys.append("{0}_count".format(s))
            self.feature_values.append(cnt)

        ratio = 0.0
        cnt = 0.0
        if 'browser' in counts:
            cnt = counts['browser']
        ratio = cnt / (cnt + counts['server'])
        self.feature_keys.append('browser_server_ratio')
        self.feature_values.append(ratio)
        return self

    def extract_hour_count(self):
        access_hours = list(set([int(log['time'].strftime('%H')) for log in self.logs]))
        x = len(access_hours) + 0.0
        self.feature_keys.append('request_hour_count')
        self.feature_values.append(x)

        mean = sum(access_hours) / x
        self.feature_keys.append('request_hour_mean')
        self.feature_values.append(mean)

        std = reduce(lambda x, y: x + y, [(h - mean)**2 for h in access_hours])
        std = math.sqrt(std / x)
        self.feature_keys.append('request_hour_std')
        self.feature_values.append(std)
        return self

    def extract_hour_allocate(self):
        access_hours = [int(log['time'].strftime('%H')) for log in self.logs]
        counter = Counter(access_hours)

        hits = [0]*24
        for h in range(0, 24):
            if h in counter:
                hits[h] = counter[h]

        top = argmax_index(hits)
        self.feature_keys.append('top_request_hour')
        self.feature_values.append(top)

        # 0am - 6am
        cnt = 0.0
        for h in range(7):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append("request_hour_0_6am")
        self.feature_values.append(cnt / len(access_hours))

        # 6am - 9am
        cnt = 0.0
        for h in range(6, 10):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_6_9am')
        self.feature_values.append(cnt / len(access_hours))

        # 8-12am
        cnt = 0.0
        for h in range(8, 13):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_8_12am')
        self.feature_values.append(cnt / len(access_hours))

        # 12 - 18pm
        cnt = 0.0
        for h in range(12, 19):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_12_18pm')
        self.feature_values.append(cnt / len(access_hours))

        # 17 - 20pm
        cnt = 0.0
        for h in range(17, 21):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_17_20pm')
        self.feature_values.append(cnt / len(access_hours))

        # 19 - 24pm
        cnt = 0.0
        for h in range(19, 25):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_19_24pm')
        self.feature_values.append(cnt / len(access_hours))
        return self

    def extract_session_count(self, unit = 1):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        session_events = {}
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= unit * 60 * 60:
                    cnt += 1.0
                    total_count += 1.0

                session_events.setdefault(total_count, []).append(i)
            cnt += 1.0
            total_count += 1.0
            session_events.setdefault(total_count, []).append(len(v))
            sessions.append(cnt)
        self.feature_keys.append('%dh_session_count' % unit)
        self.feature_values.append(total_count)

        event_cnt = [len(v) for k, v in session_events.items()]
        self.feature_keys.append('max_event_count_per_%dh_session' % unit)
        self.feature_values.append(max(event_cnt))
        self.feature_keys.append('min_event_count_per_%dh_session' % unit)
        self.feature_values.append(min(event_cnt))
        self.feature_keys.append('mean_event_count_per_%dh_session' % unit)
        mean = sum(event_cnt) / (len(event_cnt) + 0.0)
        self.feature_values.append(mean)
        std = 0.0
        for cnt in event_cnt:
            std += (cnt - mean)**2
        std = math.sqrt(std / (len(event_cnt) + 0.0))
        self.feature_keys.append('std_event_count_per_%dh_session' % unit)
        self.feature_values.append(std)


        return self

    def extract_session_lstweek(self, unit=1, w=1):
        start = self.end_day - datetime.timedelta(days=7*w)
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'] >= start]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        session_events = {}
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 60 * unit:
                    cnt += 1.0
                    total_count += 1.0
                session_events.setdefault(total_count, []).append(i)
            cnt += 1.0
            total_count += 1.0
            session_events.setdefault(total_count, []).append(len(v))
            sessions.append(cnt)
        self.feature_keys.append('%dhsession_count_lst%dweek' % (unit, w))
        self.feature_values.append(total_count)

        event_cnt = [len(v) for k, v in session_events.items()]
        self.feature_keys.append('max_event_count_per_%dh_session_lst%dweek' % (unit, w))
        self.feature_values.append(max(event_cnt))
        self.feature_keys.append('min_event_count_per_%dh_session_lst%dweek' % (unit, w))
        self.feature_values.append(min(event_cnt))
        mean = sum(event_cnt) / (len(event_cnt) + 0.0)
        self.feature_keys.append('mean_event_count_per_%dh_session_lst%dweek' % (unit, w))
        self.feature_values.append(mean)

        std = 0.0
        for cnt in event_cnt:
            std += (cnt - mean)**2
        std = math.sqrt(std / (len(event_cnt) + 0.0))
        self.feature_keys.append('std_event_count_per_%dh_session_lst%dweek' % (unit, w))
        self.feature_values.append(std)
        return self


    def extract_session_per_day(self, unit=1):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 60 * unit:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('%dh_session_per_day' % unit)
        self.feature_values.append(total_count / (len(sessions) + 0.0))

        return self


    def extract_request_lag(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            lags = [999] # missing value
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        self.feature_keys.append('min_request_lag')
        self.feature_values.append(min(lags) + 0.0)

        self.feature_keys.append('max_request_lag')
        self.feature_values.append(max(lags) + 0.0)

        mean = sum(lags) / (len(lags) + 0.0)
        self.feature_keys.append('mean_request_lag')
        self.feature_values.append(mean)

        std = 0.0
        for l in lags:
            std += (l - mean)**2
        std = math.sqrt(std / (len(lags) + 0.0))
        self.feature_keys.append('std_request_lag')
        self.feature_values.append(std)
        return self


    def extract_request_lags(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        lags = []
        if len(request_dates) >= 2:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        sum_lag = sum(lags) + 0.0
        self.feature_keys.append('total_lag')
        self.feature_values.append(sum_lag + .0)

        x2d = 0.0
        x2d_ratio = 0.0
        x3d = 0.0
        x3d_ratio = 0.0
        x5d = 0.0
        x5d_ratio = 0.0
        x7d = 0.0
        x7d_ratio = 0.0

        for lag in lags:
            if lag <= 2:
                x2d += 1.0
            if lag >= 3 and lag < 5:
                x3d += 1.0
            if lag >= 5 and lag < 8:
                x5d += 1.0
            if lag >= 8:
                x7d += 1.0

        if len(lags) > 0:
            n = len(lags) + 0.0
            x2d_ratio = x2d / n
            x3d_ratio = x3d / n
            x5d_ratio = x5d / n
            x7d_ratio = x7d / n

        self.feature_keys.append('lag<2d_times')
        self.feature_values.append(x2d)
        self.feature_keys.append('lag<2d_ratio')
        self.feature_values.append(x2d_ratio)
        self.feature_keys.append('lag>3d_times')
        self.feature_values.append(x3d)
        self.feature_keys.append('lag>3d_ratio')
        self.feature_values.append(x3d_ratio)
        self.feature_keys.append('lag>5d_times')
        self.feature_values.append(x5d)
        self.feature_keys.append('lag>5d_ratio')
        self.feature_values.append(x5d_ratio)
        self.feature_keys.append('lag>7d_times')
        self.feature_values.append(x7d)
        self.feature_keys.append('lag>7d_ratio')
        self.feature_values.append(x7d_ratio)


        self.feature_keys.append('sumlag<1week')
        self.feature_keys.append('sumlag1<>2week')
        self.feature_keys.append('sumlag>2week')
        if sum_lag == 0:
            self.feature_values.append(1)
            self.feature_values.append(1)
            self.feature_values.append(1)
        elif sum_lag < 7:
            self.feature_values.append(1)
            self.feature_values.append(0)
            self.feature_values.append(0)
        elif sum_lag >= 7 and sum_lag < 14:
            self.feature_values.append(0)
            self.feature_values.append(1)
            self.feature_values.append(0)
        else:
            self.feature_values.append(0)
            self.feature_values.append(0)
            self.feature_values.append(1)

        return self


    def extract_staytime(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        daily_logs = {}
        for d, t in requests:
            daily_logs.setdefault(d, []).append(t)
        staytimes = [(max(v)-min(v)).seconds/60.0 for k, v in daily_logs.items()]

        self.feature_keys.append('total_staytime')
        self.feature_values.append(sum(staytimes) / 60.0)

        self.feature_keys.append('staytime_max')
        self.feature_values.append(max(staytimes))
        self.feature_keys.append('staytime_min')
        self.feature_values.append(min(staytimes))
        self.feature_keys.append('staytime_mean')
        mean = sum(staytimes) / (len(staytimes) + 0.0)
        self.feature_values.append(mean)

        std = reduce(lambda x, y: x+y, [(s - mean)**2 for s in staytimes]) / (len(staytimes) + 0.0)
        std = math.sqrt(std)

        self.feature_keys.append('staytime_std')
        self.feature_values.append(std)

        m = median(staytimes)
        self.feature_keys.append('staytime_median')
        self.feature_values.append(m)

        p25 = percentile(staytimes, 0.25)
        p75 = percentile(staytimes, 0.75)
        self.feature_keys.append('staytime_25p')
        self.feature_values.append(p25)
        self.feature_keys.append('staytime_75p')
        self.feature_values.append(p75)
        return self

    def extract_staytime_lstweek(self, w=1):
        start = self.end_day - datetime.timedelta(days=7*w)
        requests = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'] >= start]
        daily_logs = {}
        for d, t in requests:
            daily_logs.setdefault(d, []).append(t)
        staytimes = [(max(v)-min(v)).seconds/60.0 for k, v in daily_logs.items()]

        self.feature_keys.append('total_staytime_lst%dweek' % w)
        self.feature_values.append(sum(staytimes) / 60.0)

        self.feature_keys.append('staytime_max_lst%dweek' % w)
        self.feature_values.append(max(staytimes))
        self.feature_keys.append('staytime_min_lst%dweek' % w)
        self.feature_values.append(min(staytimes))
        self.feature_keys.append('staytime_mean_lst%dweek' % w)
        mean = sum(staytimes) / (len(staytimes) + 0.0)
        self.feature_values.append(mean)

        std = reduce(lambda x, y: x+y, [(s - mean)**2 for s in staytimes]) / (len(staytimes) + 0.0)
        std = math.sqrt(std)

        self.feature_keys.append('staytime_std_lst%dweek' % w)
        self.feature_values.append(std)

        m = median(staytimes)
        self.feature_keys.append('staytime_median_lst%dweek' % w)
        self.feature_values.append(m)

        p25 = percentile(staytimes, 0.25)
        p75 = percentile(staytimes, 0.75)
        self.feature_keys.append('staytime_25p_lst%dweek' % w)
        self.feature_values.append(p25)
        self.feature_keys.append('staytime_75p_lst%dweek' % w)
        self.feature_values.append(p75)
        return self


    def extract_problem_over3minutes_count(self):
        cnt = 0
        for i in xrange(len(self.logs) - 1):
            if self.logs[i]['event'] != 'problem':
                continue

            time_delta = self.logs[i+1]['time'] - self.logs[i]['time']

            if 180 <= time_delta.seconds <= 18000 and time_delta.days == 0:
                cnt += 1.0
        self.feature_keys.append('problem_over3minutes_count')
        self.feature_values.append(cnt)
        return self

    def extract_request_weekend_count(self):
        ws = [log['time'].weekday() for log in self.logs]
        d = len([1 for w in ws if w > 5]) + 0.0
        self.feature_keys.append('request_weekend_count')
        self.feature_values.append(d)

        if d > (len(ws) - d):
            self.feature_keys.append('more_weekend')
            self.feature_values.append(1)
            self.feature_keys.append('more_workday')
            self.feature_values.append(0)
        else:
            self.feature_keys.append('more_weekend')
            self.feature_values.append(0)
            self.feature_keys.append('more_workday')
            self.feature_values.append(1)

        return self


    def extract_request_weekend_percentage(self):
        ws = [log['time'].weekday() for log in self.logs]
        self.feature_keys.append('request_weekend_percentage')
        self.feature_values.append(float(sum([1 for w in ws if w > 5])) / len(ws))
        return self


    def extract_video_over10minutes_count(self):
        cnt_10m = 0
        cnt_30m = 0

        for i in xrange(len(self.logs) - 1):
            if self.logs[i]['event'] != 'video':
                continue

            time_delta = self.logs[i+1]['time'] - self.logs[i]['time']

            if 600 <= time_delta.seconds < 1800 and time_delta.days == 0:
                cnt_10m += 1.0
            if 1800 <= time_delta.seconds <= 18000 and time_delta.days== 0:
                cnt_30m += 1.0
        self.feature_keys.append('video_over10minutes_count')
        self.feature_values.append(cnt_10m)

        self.feature_keys.append('video_over30minutes_count')
        self.feature_values.append(cnt_30m)

        videos = [log['object'] for log in self.logs if log['event'] == 'video']
        n = len(videos) + 0.0

        ratio = 1.0
        ratio_10m = 0.0
        ratio_30m = 0.0

        if n > 0:
            ratio = (n - cnt_10m - cnt_30m) / n
            ratio_10m = cnt_10m / n
            ratio_30m = cnt_30m / n
        self.feature_keys.append('video_less10minutes_count')
        self.feature_values.append(n - cnt_10m - cnt_30m)

        self.feature_keys.append('video_less10minutes_ratio')
        self.feature_values.append(ratio)
        self.feature_keys.append('video_over10minutes_ratio')
        self.feature_values.append(ratio_10m)
        self.feature_keys.append('video_over30minutes_ratio')
        self.feature_values.append(ratio_30m)

        return self


    def extract_daytime(self):
        request_dates = [int(log['time'].strftime('%H')) for log in self.logs]
        counts = {}
        day_times = 0.0
        night_times = 0.0
        for h in request_dates:
            # if h not in counts:
            #     counts[h] = 1.0
            # else:
            #     counts[h] += 1.0
            if h < 19 and h >= 7:
                day_times += 1.0
            else:
                night_times += 1.0


        self.feature_keys.append('daytime_ratio')
        self.feature_values.append(day_times / (day_times + night_times))

        if night_times >= day_times:
            self.feature_keys.append('day_time')
            self.feature_values.append(0)

            self.feature_keys.append('night_time')
            self.feature_values.append(1)

        else:
            self.feature_keys.append('day_time')
            self.feature_values.append(1)

            self.feature_keys.append('night_time')
            self.feature_values.append(0)
        return self

    def extract_moduel_problem(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['event']) for log in self.logs]
        events = {}
        for d, e in requests:
            events.setdefault(d, []).append(e)
        for k, v in events.items():
            events[k] = set(v)

        cnt = 0.0
        access_cnt = 0.0
        for k, v in events.items():
            if 'access' in v and 'problem' in v:
                cnt += 1.0
            if 'access' in v:
                access_cnt += 1.0
        self.feature_keys.append('module_and_problem_count')
        self.feature_values.append(cnt)

        self.feature_keys.append('module_problem_ratio')
        ratio = 0.0
        if access_cnt > 0:
            ratio = cnt / access_cnt
        self.feature_values.append(ratio)

        cnt = 0.0
        prob_cnt = 0.0
        for k, v in events.items():
            if 'problem' in v and 'access' not in v:
                cnt += 1.0
            if 'problem' in v:
                prob_cnt += 1.0
        ratio = 0.0
        if prob_cnt > 0:
            ratio = cnt / prob_cnt

        self.feature_keys.append('module_not_problem_count')
        self.feature_values.append(cnt)

        self.feature_keys.append('module_not_problem_ratio')
        self.feature_values.append(ratio)
        return self
