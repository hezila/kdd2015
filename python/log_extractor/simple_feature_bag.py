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


    def extract_duration_days(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        duration = (request_dates[-1] - request_dates[0]).days
        if duration == 0:
            duration = 1.0

        self.feature_keys.append('duration_log')
        self.feature_values.append(count_log(duration))

        return self

    def extract_request_count(self):
        # ignore the page close event
        requests = [log for log in self.logs if log['event'] != 'page_close']
        x = len(requests) + 0.0
        self.feature_keys.append("request_count_log")
        self.feature_values.append(count_log(x))
        return self

    def extract_request_count_lst2week(self):
        # ignore the page close event
        start = self.end_day - datetime.timedelta(days=14)
        requests = [log for log in self.logs if (log['event'] != 'page_close' and log['time'] >= start)]
        x = len(requests) + 0.0
        self.feature_keys.append("request_count_lst2week_log")
        self.feature_values.append(count_log(x))
        return self

    def extract_request_count_lst1week(self):
        # ignore the page close event
        start = self.end_day - datetime.timedelta(days=7)
        requests = [log for log in self.logs if (log['event'] != 'page_close' and log['time'] >= start)]
        x = len(requests) + 0.0
        self.feature_keys.append("request_count_lst1week_log")
        self.feature_values.append(count_log(x))
        return self


    def extract_event_count(self):
        x = len(self.logs) + 0.0
        self.feature_keys.append("event_count_log")
        self.feature_values.append(count_log(x))
        return self

    def extract_event_count_lst2week(self):
        start = self.end_day - datetime.timedelta(days=14)
        requests = [log for log in self.logs if log['time'] >= start]
        x = len(requests) + 0.0
        self.feature_keys.append("event_count_lst2week_log")
        self.feature_values.append(math.log(x))
        return self

    def extract_event_count_lst1week(self):
        start = self.end_day - datetime.timedelta(days=7)
        requests = [log for log in self.logs if log['time'] >= start]
        x = len(requests) + 0.0
        self.feature_keys.append("event_count_lst1week_log")
        self.feature_values.append(count_log(x))
        return self

    def extract_active_days(self):
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs])
        days = len(request_dates) + 0.0

        self.feature_keys.append('active_days_log')
        self.feature_values.append(count_log(days))

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

        self.feature_keys.append('max_days_per_week_log')
        self.feature_values.append(count_log(max(x)))

        self.feature_keys.append('min_days_per_week_log')
        self.feature_values.append(count_log(min(x)))

        self.feature_keys.append('mean_days_per_week_log')
        self.feature_values.append(count_log(sum(x) / (len(x) + 0.0)))


        duration = (self.end_day - self.start_day).days
        if duration == 0:
            duration = 1.0
        x = days / duration
        self.feature_keys.append('active_days_freq_log')
        self.feature_values.append(count_log(x))
        return self

    def extract_active_days_per_week(self):
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs])
        d = len(request_dates) + 0.0

        w = (self.end_day - self.start_day).days / 7.0
        if w == 0:
            w = 1.0
        x = d / w
        self.feature_keys.append('active_days_per_week_log')
        self.feature_values.append(count_log(x))
        return self

    def extract_active_days_lst2week(self):
        start = self.end_day - datetime.timedelta(days=14)
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs if log['time'] >= start])
        days = len(request_dates) + 0.0

        self.feature_keys.append('active_days_lst2week_log')
        self.feature_values.append(count_log(days))

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

        self.feature_keys.append('max_days_lst2week_log')
        self.feature_values.append(count_log(max([x1, x2])))

        self.feature_keys.append('min_days_lst2week_log')
        self.feature_values.append(count_log(min([x1, x2])))

        self.feature_keys.append('mean_days_lst2week_log')
        self.feature_values.append(count_log((x1 + x2) / 2.0))

        return self

    def extract_active_days_lst1week(self):
        start = self.end_day - datetime.timedelta(days=7)
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs if log['time'] >= start])
        days = len(request_dates) + 0.0

        self.feature_keys.append('active_days_lst1week_log')
        self.feature_values.append(count_log(days))
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
            self.feature_keys.append('server_{0}_count_log'.format(e))
            self.feature_values.append(count_log(cnt))
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
            self.feature_keys.append('server_{0}_count_lst2week_log'.format(e))
            self.feature_values.append(count_log(cnt))
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
            self.feature_keys.append('server_{0}_count_lst1week_log'.format(e))
            self.feature_values.append(count_log(cnt))
        return self

    def extract_access_count(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if log['event'] == 'access']
        self.feature_keys.append('access_count_log')
        x = len(requests) + 0.0
        self.feature_values.append(count_log(x))

        daily_access = {}
        for d, t in requests:
            daily_access.setdefault(d, []).append(t)

        modules = []
        for d, vs in daily_access.items():
            for v in vs:
                if v not in modules:
                    modules.append(v)
        self.feature_keys.append('module_count_log')
        self.feature_values.append(count_log(len(modules)))

        modules = []
        for d, vs in daily_access.items():
            modules.append(len(set(vs)))

        max_m = 0
        median_m = 0
        if len(modules) > 0:
            max_m = max(modules)
            median_m = median(modules)
        self.feature_keys.append('max_module_per_day_log')
        self.feature_values.append(count_log(max_m))
        #self.feature_keys.append('min_module_per_day')
        #self.feature_values.append(min(modules))
        self.feature_keys.append('median_module_per_day_log')
        self.feature_values.append(count_log(median_m))

        avg = 0.0
        if len(modules) > 0:
            avg = sum(modules) / (len(modules) + 0.0)
        self.feature_keys.append('mean_module_per_day_log')
        self.feature_values.append(count_log(avg))

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
        self.feature_keys.append('access_count_lst%dweek_log' % w)
        self.feature_values.append(count_log(len(requests)))

        daily_access = {}
        for d, t in requests:
            daily_access.setdefault(d, []).append(t)

        modules = []
        for d, vs in daily_access.items():
            for v in vs:
                if v not in modules:
                    modules.append(v)
        self.feature_keys.append('module_count_lst%dweek_log' % w)
        self.feature_values.append(count_log(len(modules)))

        modules = []
        for d, vs in daily_access.items():
            modules.append(len(set(vs)))

        max_m = 0
        median_m = 0
        if len(modules) > 0:
            max_m = max(modules)
            median_m = median(modules)

        self.feature_keys.append('max_module_lst%dweek_log' % w)
        self.feature_values.append(count_log(max_m))

        #self.feature_keys.append('min_module_lst2week')
        #self.feature_values.append(min(modules))

        self.feature_keys.append('median_module_lst%dweek_log' % w)
        self.feature_values.append(count_log(median_m))

        avg = 0.0
        if len(modules) > 0:
            avg = sum(modules) / (len(modules) + 0.0)
        self.feature_keys.append('mean_module_lst%dweek_log' % w)
        self.feature_values.append(count_log(avg))



        return self


    def extract_video_count(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if log['event'] == 'video']
        self.feature_keys.append('video_request_count_log')
        self.feature_values.append(count_log(len(requests)))

        daily_videos = {}
        for d, t in requests:
            daily_videos.setdefault(d, []).append(t)

        videos = []
        for d, vs in daily_videos.items():
            for v in vs:
                if v not in videos:
                    videos.append(v)
        self.feature_keys.append('video_count_log')
        self.feature_values.append(count_log(len(videos)))

        self.feature_keys.append('video_perf_log')
        self.feature_values.append(count_log(len(requests) / (len(videos) + 0.1)))

        videos = []
        for d, vs in daily_videos.items():
            videos.append(len(set(vs)))

        max_v = 0
        median_v = 0
        if len(videos) > 0:
            max_v = max(videos)
            median_v = median(videos)

        self.feature_keys.append('max_video_per_day_log')
        self.feature_values.append(count_log(max_v))
        self.feature_keys.append('median_video_per_day_log')
        self.feature_values.append(count_log(median_v))


        avg = 0.0
        if len(videos) > 0:
            avg = sum(videos) / (len(videos) + 0.0)
        self.feature_keys.append('mean_video_per_day_log')
        self.feature_values.append(count_log(avg))

        return self


    def extract_video_count_lstweek(self, w):
        start = self.end_day - datetime.timedelta(days=7 * w)
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if log['event'] == 'video' and log['time'] >= start]
        self.feature_keys.append('video_request_count_lst%dweek_log' % w)
        self.feature_values.append(count_log(len(requests)))

        daily_videos = {}
        for d, t in requests:
            daily_videos.setdefault(d, []).append(t)

        videos = []
        for d, vs in daily_videos.items():
            for v in vs:
                if v not in videos:
                    videos.append(v)
        self.feature_keys.append('video_count_lst%dweek_log' % w)
        self.feature_values.append(count_log(len(videos)))

        videos = []
        for d, vs in daily_videos.items():
            videos.append(len(set(vs)))

        max_v = 0
        median_v = 0
        if len(videos) > 0:
            max_v = max(videos)
            median_v = median(videos)

        self.feature_keys.append('max_videos_perday_lst%dweek_log' % w)
        self.feature_values.append(count_log(max_v))
        self.feature_keys.append('median_videos_perday_lst%dweek_log' % w)
        self.feature_values.append(count_log(median_v))


        avg = 0.0
        if len(videos) > 0:
            avg = sum(videos) / (len(videos) + 0.0)
        self.feature_keys.append('mean_video_perday_lst%dweek_log' % w)
        self.feature_values.append(count_log(avg))

        return self


    def extract_problem_count(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if log['event'] == 'problem']
        self.feature_keys.append('problem_request_count_log')
        self.feature_values.append(count_log(len(requests)))

        daily_problems = {}
        for d, t in requests:
            daily_problems.setdefault(d, []).append(t)

        problems = []
        for d, vs in daily_problems.items():
            for v in vs:
                if v not in problems:
                    problems.append(v)

        self.feature_keys.append('problem_count_log')
        self.feature_values.append(count_log(len(problems)))

        self.feature_keys.append('problem_perf_log')
        self.feature_values.append(count_log(len(requests) / (len(problems) + 0.1)))

        problems = []
        for d, vs in daily_problems.items():
            problems.append(len(set(vs)))
        max_p = 0
        median_p = 0
        if len(problems) > 0:
            max_p = max(problems)
            median_p = median(problems)

        self.feature_keys.append('max_problem_per_day_log')
        self.feature_values.append(count_log(max_p))

        self.feature_keys.append('median_problem_per_day_log')
        self.feature_values.append(count_log(median_p))

        avg = 0.0
        if len(problems) > 0:
            avg = sum(problems) / (len(problems) + 0.0)
        self.feature_keys.append('mean_problem_per_day_log')
        self.feature_values.append(count_log(avg))
        return self

    def extract_problem_count_lst2week(self):
        start = self.end_day - datetime.timedelta(days=14)
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if (log['event'] == 'problem' and log['time'] >= start)]
        self.feature_keys.append('problem_request_count_lst2week_log')
        self.feature_values.append(count_log(len(requests)))

        daily_problems = {}
        for d, t in requests:
            daily_problems.setdefault(d, []).append(t)

        problems = []
        for d, vs in daily_problems.items():
            for v in vs:
                if v not in problems:
                    problems.append(v)

        self.feature_keys.append('problem_count_lst2week_log')
        self.feature_values.append(count_log(len(problems)))

        problems = []
        for d, vs in daily_problems.items():
            problems.append(len(set(vs)))
        max_p = 0
        median_p = 0
        if len(problems) > 0:
            max_p = max(problems)
            median_p = median(problems)
        self.feature_keys.append('max_problem_per_day_lst2week_log')
        self.feature_values.append(count_log(max_p))
        self.feature_keys.append('median_problem_per_day_lst2week_log')
        self.feature_values.append(count_log(median_p))

        avg = 0.0
        if len(problems) > 0:
            avg = sum(problems) / (len(problems) + 0.0)
        self.feature_keys.append('problem_per_day_lst2week_log')
        self.feature_values.append(count_log(avg))
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
            self.feature_keys.append('browser_{0}_count_log'.format(e))
            self.feature_values.append(count_log(cnt))
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
            self.feature_keys.append('browser_{0}_count_lst2week_log'.format(e))
            self.feature_values.append(count_log(cnt))
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
            self.feature_keys.append('browser_{0}_count_lst1week_log'.format(e))
            self.feature_values.append(count_log(cnt))
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
            self.feature_keys.append("{0}_count_log".format(s))
            self.feature_values.append(count_log(cnt))


        return self

    def extract_hour_count(self):
        access_hours = list(set([int(log['time'].strftime('%H')) for log in self.logs]))
        x = len(access_hours) + 0.0
        self.feature_keys.append('request_hour_count_log')
        self.feature_values.append(count_log(x))

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
        self.feature_keys.append('%dh_session_count_log' % unit)
        self.feature_values.append(count_log(total_count))

        event_cnt = [len(v) for k, v in session_events.items()]
        self.feature_keys.append('max_event_count_per_%dh_session_log' % unit)
        self.feature_values.append(count_log(max(event_cnt)))
        self.feature_keys.append('min_event_count_per_%dh_session_log' % unit)
        self.feature_values.append(count_log(min(event_cnt)))
        self.feature_keys.append('mean_event_count_per_%dh_session_log' % unit)
        mean = sum(event_cnt) / (len(event_cnt) + 0.0)
        self.feature_values.append(count_log(mean))


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
        self.feature_keys.append('%dhsession_count_lst%dweek_log' % (unit, w))
        self.feature_values.append(count_log(total_count))

        event_cnt = [len(v) for k, v in session_events.items()]
        self.feature_keys.append('max_event_count_per_%dh_session_lst%dweek_log' % (unit, w))
        self.feature_values.append(count_log(max(event_cnt)))
        self.feature_keys.append('min_event_count_per_%dh_session_lst%dweek_log' % (unit, w))
        self.feature_values.append(count_log(min(event_cnt)))
        mean = sum(event_cnt) / (len(event_cnt) + 0.0)
        self.feature_keys.append('mean_event_count_per_%dh_session_lst%dweek_log' % (unit, w))
        self.feature_values.append(count_log(mean))

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
        self.feature_keys.append('%dh_session_per_day_log' % unit)
        self.feature_values.append(count_log(total_count / (len(sessions) + 0.0)))

        return self


    def extract_request_lag(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            lags = [999] # missing value
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        self.feature_keys.append('min_request_lag_log')
        self.feature_values.append(count_log(min(lags)))

        self.feature_keys.append('max_request_lag_log')
        self.feature_values.append(count_log(max(lags)))

        mean = sum(lags) / (len(lags) + 0.0)
        self.feature_keys.append('mean_request_lag_log')
        self.feature_values.append(count_log(mean))

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

        self.feature_keys.append('lag<2d_times_log')
        self.feature_values.append(count_log(x2d))
        self.feature_keys.append('lag>3d_times_log')
        self.feature_values.append(count_log(x3d))

        self.feature_keys.append('lag>5d_times_log')
        self.feature_values.append(count_log(x5d))

        self.feature_keys.append('lag>7d_times_log')
        self.feature_values.append(count_log(x7d))
        return self


    def extract_staytime(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        daily_logs = {}
        for d, t in requests:
            daily_logs.setdefault(d, []).append(t)
        staytimes = [(max(v)-min(v)).seconds/60.0 for k, v in daily_logs.items()]

        self.feature_keys.append('total_staytime_log')
        self.feature_values.append(count_log(sum(staytimes) / 60.0))

        self.feature_keys.append('staytime_max_log')
        self.feature_values.append(count_log(max(staytimes)))
        self.feature_keys.append('staytime_min_log')
        self.feature_values.append(count_log(min(staytimes)))
        self.feature_keys.append('staytime_mean_log')
        mean = sum(staytimes) / (len(staytimes) + 0.0)
        self.feature_values.append(count_log(mean))

        return self

    def extract_staytime_lstweek(self, w=1):
        start = self.end_day - datetime.timedelta(days=7*w)
        requests = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'] >= start]
        daily_logs = {}
        for d, t in requests:
            daily_logs.setdefault(d, []).append(t)
        staytimes = [(max(v)-min(v)).seconds/60.0 for k, v in daily_logs.items()]

        self.feature_keys.append('total_staytime_lst%dweek_log' % w)
        self.feature_values.append(count_log(sum(staytimes) / 60.0))

        self.feature_keys.append('staytime_max_lst%dweek_log' % w)
        self.feature_values.append(count_log(max(staytimes)))
        self.feature_keys.append('staytime_min_lst%dweek_log' % w)
        self.feature_values.append(count_log(min(staytimes)))
        self.feature_keys.append('staytime_mean_lst%dweek_log' % w)
        mean = sum(staytimes) / (len(staytimes) + 0.0)
        self.feature_values.append(count_log(mean))


        return self


    def extract_problem_over3minutes_count(self):
        cnt = 0
        for i in xrange(len(self.logs) - 1):
            if self.logs[i]['event'] != 'problem':
                continue

            time_delta = self.logs[i+1]['time'] - self.logs[i]['time']

            if 180 <= time_delta.seconds <= 18000 and time_delta.days == 0:
                cnt += 1.0
        self.feature_keys.append('problem_over3minutes_count_log')
        self.feature_values.append(count_log(cnt))
        return self

    def extract_request_weekend_count(self):
        ws = [log['time'].weekday() for log in self.logs]
        d = len([1 for w in ws if w > 5]) + 0.0
        self.feature_keys.append('request_weekend_count_log')
        self.feature_values.append(count_log(d))


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
        self.feature_keys.append('video_over10minutes_count_log')
        self.feature_values.append(count_log(cnt_10m))

        self.feature_keys.append('video_over30minutes_count_log')
        self.feature_values.append(count_log(cnt_30m))


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
        self.feature_keys.append('module_and_problem_count_log')
        self.feature_values.append(count_log(cnt))

        # self.feature_keys.append('module_problem_ratio')
        ratio = 0.0
        if access_cnt > 0:
            ratio = cnt / access_cnt
        # self.feature_values.append(ratio)

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

        self.feature_keys.append('module_not_problem_count_log')
        self.feature_values.append(count_log(cnt))

        
        return self
