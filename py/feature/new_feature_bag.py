#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import Counter
import datetime
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

class NewFeatureBag(FeatureBag):
    def __init__(self, enrollment_id, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)
        self.logs = sorted(logs, key=lambda x: x['time'])
        self.start = self.logs[0]['time'].strftime('%Y%m%d')
        self.start_day = datetime.datetime.strptime(self.start, '%Y%m%d')

        self.end = self.logs[-1]['time'].strftime('%Y%m%d')
        self.end_day = datetime.datetime.strptime(self.end, '%Y%m%d')




    def extract_request_count_lstday(self):
        # ignore the page close event
        requests = [log for log in self.logs if (log['event'] != 'page_close' and log['time'].strftime('%Y%m%d') == self.end)]
        x = len(requests) + 0.0
        self.feature_keys.append("request_count_lstday")
        self.feature_values.append(x)
        return self


    def extract_event_count_lstday(self):
        requests = [log for log in self.logs if (log['time'].strftime('%Y%m%d') == self.end)]

        x = len(requests) + 0.0
        self.feature_keys.append("event_count_lstday")
        self.feature_values.append(x)
        return self


    def extract_lst_lag(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))

        lag = 14
        if len(request_dates) > 1:
            lag = (datetime.datetime.strptime(request_dates[-2], '%Y%m%d') - self.end_day).days


        self.feature_keys.append('lst_lag')
        self.feature_values.append(lag)


        return self

    # TODO: active days >= 30 minutes

    def extract_server_events_lstday(self):
        events = [log['event'] for log in self.logs if (log['source'] == 'server' and log['time'].strftime('%Y%m%d') == self.end)]
        counts = {}
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1
        for e in server_events:
            cnt = 0.0
            if e in counts:
                cnt = counts[e]
            self.feature_keys.append('server_{0}_count_lstday'.format(e))
            self.feature_values.append(cnt)
        return self


    def extract_access_count_lstday(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if (log['event'] == 'access' and log['time'].strftime('%Y%m%d') == self.end)]
        self.feature_keys.append('request_module_count_lstday')
        self.feature_values.append(len(requests) + 0.0)

        daily_access = {}
        for d, t in requests:
            daily_access.setdefault(d, []).append(t)

        modules = []
        for d, vs in daily_access.items():
            for v in vs:
                if v not in modules:
                    modules.append(v)
        self.feature_keys.append('module_count_lstday')
        self.feature_values.append(len(modules) + 0.0)


        return self



    def extract_video_count_lstday(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if (log['event'] == 'video' and log['time'].strftime('%Y%m%d') == self.end)]
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
        self.feature_keys.append('video_count_lstday')
        self.feature_values.append(len(videos) + 0.0)


        return self



    def extract_problem_count_lstday(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['object']) for log in self.logs if (log['event'] == 'problem' and log['time'].strftime('%Y%m%d') == self.end)]
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

        self.feature_keys.append('problem_count_lstday')
        self.feature_values.append(len(problems) + 0.0)


        return self



    def extract_browser_events_lstday(self):
        events = [log['event'] for log in self.logs if (log['source'] == 'browser' and log['time'].strftime('%Y%m%d') == self.end)]
        counts = {}
        for e in events:
            counts.setdefault(e, 0.0)
            counts[e] += 1
        for e in browser_events:
            cnt = 0.0
            if e in counts:
                cnt = counts[e]
            self.feature_keys.append('browser_{0}_count_lstday'.format(e))
            self.feature_values.append(cnt)
        return self


    def extract_source_count_lstday(self):
        events = [log['source'] for log in self.logs if log['time'].strftime('%Y%m%d') == self.end]
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
            self.feature_keys.append("{0}_count_lstday".format(s))
            self.feature_values.append(cnt)

        ratio = 0.0
        cnt = 0.0
        if 'browser' in counts:
            cnt = counts['browser']
        ratio = cnt / (cnt + counts['server'])
        self.feature_keys.append('browser_server_ratio_lstday')
        self.feature_values.append(ratio)
        return self

    def extract_hour_count_lstday(self):
        access_hours = list(set([int(log['time'].strftime('%H')) for log in self.logs if log['time'].strftime('%Y%m%d') == self.end]))
        x = len(access_hours) + 0.0
        self.feature_keys.append('request_hour_count_lstday')
        self.feature_values.append(x)

        mean = sum(access_hours) / x
        self.feature_keys.append('request_hour_mean_lstday')
        self.feature_values.append(mean)

        std = reduce(lambda x, y: x + y, [(h - mean)**2 for h in access_hours])
        std = math.sqrt(std / x)
        self.feature_keys.append('request_hour_std_lstday')
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

    def extract_hour_allocate_lstday(self):
        access_hours = [int(log['time'].strftime('%H')) for log in self.logs if log['time'].strftime('%Y%m%d') == self.end]
        counter = Counter(access_hours)

        hits = [0]*24
        for h in range(0, 24):
            if h in counter:
                hits[h] = counter[h]

        top = argmax_index(hits)
        self.feature_keys.append('top_request_hour_lstday')
        self.feature_values.append(top)

        # 0am - 6am
        cnt = 0.0
        for h in range(7):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append("request_hour_0_6am_lstday")
        self.feature_values.append(cnt / len(access_hours))

        # 6am - 9am
        cnt = 0.0
        for h in range(6, 10):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_6_9am_lstday')
        self.feature_values.append(cnt / len(access_hours))

        # 8-12am
        cnt = 0.0
        for h in range(8, 13):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_8_12am_lstday')
        self.feature_values.append(cnt / len(access_hours))

        # 12 - 18pm
        cnt = 0.0
        for h in range(12, 19):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_12_18pm_lstday')
        self.feature_values.append(cnt / len(access_hours))

        # 17 - 20pm
        cnt = 0.0
        for h in range(17, 21):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_17_20pm_lstday')
        self.feature_values.append(cnt / len(access_hours))

        # 19 - 24pm
        cnt = 0.0
        for h in range(19, 25):
            if h in counter:
                cnt += counter[h]
        self.feature_keys.append('request_hour_19_24pm_lstday')
        self.feature_values.append(cnt / len(access_hours))
        return self


    def extract_session_count_2h(self):
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
                if (v[i+1] - v[i]).seconds >= 60 * 60 * 2:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_count_2h')
        self.feature_values.append(total_count)

        return self

    def extract_session_count_lstday_2h(self):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'].strftime('%Y%m%d') == self.end]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 60 * 2:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_count_lstday_2h')
        self.feature_values.append(total_count)

        return self


    def extract_session_count_3h(self):
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
                if (v[i+1] - v[i]).seconds >= 60 * 60 * 3:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_count_3h')
        self.feature_values.append(total_count)

        return self

    def extract_session_count_lstday_3h(self):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'].strftime('%Y%m%d') == self.end]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 60 * 3:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_count_lstday_3h')
        self.feature_values.append(total_count)

        return self

    def extract_session_lst2week_2h(self):
        start = self.end_day - datetime.timedelta(days=14)
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'] >= start]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 60*2:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_count_lst2week_2h')
        self.feature_values.append(total_count)

        return self

    def extract_session_lst2week_3h(self):
        start = self.end_day - datetime.timedelta(days=14)
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'] >= start]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 60*3:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_count_lst2week_3h')
        self.feature_values.append(total_count)

        return self


    def extract_session_lst1week_2h(self):
        start = self.end_day - datetime.timedelta(days=7)
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'] >= start]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 60 * 2:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_count_lst1week_2h')
        self.feature_values.append(total_count)

        return self

    def extract_session_lst1week_3h(self):
        start = self.end_day - datetime.timedelta(days=7)
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs if log['time'] >= start]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        sessions = []
        total_count = 0.0
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 60 * 3:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_count_lst1week_3h')
        self.feature_values.append(total_count)

        return self



    def extract_session_per_day_2h(self):
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
                if (v[i+1] - v[i]).seconds >= 60 * 60 * 2:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_per_day_2h')
        self.feature_values.append(total_count / (len(sessions) + 0.0))

        return self

    def extract_session_per_day_3h(self):
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
                if (v[i+1] - v[i]).seconds >= 60 * 60 * 3:
                    cnt += 1.0
            cnt += 1.0
            total_count += cnt
            sessions.append(cnt)
        self.feature_keys.append('session_per_day_3h')
        self.feature_values.append(total_count / (len(sessions) + 0.0))

        return self




    def extract_staytime_lstday(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        daily_logs = {}
        for d, t in requests:
            daily_logs.setdefault(d, []).append(t)

        item = daily_logs[self.end]
        staytime = (max(item) - min(item)).seconds/60.0


        self.feature_keys.append('staytime_lstday')
        self.feature_values.append(staytime)
        return self

    def extract_problem_over3minutes_count_lstday(self):
        cnt = 0
        for i in xrange(len(self.logs) - 1):
            if self.logs[i]['event'] != 'problem':
                continue

            if self.logs[i]['time'].strftime('%Y%m%d') != self.end:
                continue

            time_delta = self.logs[i+1]['time'] - self.logs[i]['time']

            if 180 <= time_delta.seconds <= 18000 and time_delta.days == 0:
                cnt += 1.0
        self.feature_keys.append('problem_over3minutes_count')
        self.feature_values.append(cnt)
        return self

    def extract_request_weekend_lstday(self):
        w = self.end_day.weekday()
        if w > 5:
            self.feature_keys.append('weekend_lstday')
            self.feature_values.append(1.0)
        else:
            self.feature_keys.append('weekend_lstday')
            self.feature_values.append(0.0)

        return self





    def extract_video_over10minutes_count_lstday(self):
        cnt = 0
        for i in xrange(len(self.logs) - 1):
            if self.logs[i]['event'] != 'video':
                continue

            if self.logs[i]['time'].strftime('%Y%m%d') != self.end:
                continue

            time_delta = self.logs[i+1]['time'] - self.logs[i]['time']

            if 600 <= time_delta.seconds <= 18000 and time_delta.days == 0:
                cnt += 1.0
        self.feature_keys.append('video_over10minutes_count')
        self.feature_values.append(cnt)
        return self

    def extract_daytime_lstday(self):
        request_dates = [int(log['time'].strftime('%H')) for log in self.logs if log['time'].strftime('%Y%m%d') == self.end]
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
                night_times += 1


        self.feature_keys.append('daytime_ratio_lstday')
        self.feature_values.append(day_times / (day_times + night_times))

        if night_times >= day_times:
            self.feature_keys.append('day_time_lstday')
            self.feature_values.append(0.0)

            self.feature_keys.append('night_time_lstday')
            self.feature_values.append(1.0)

        else:
            self.feature_keys.append('day_time_lstday')
            self.feature_values.append(1.0)

            self.feature_keys.append('night_time_lstday')
            self.feature_values.append(0.0)
        return self

    def extract_moduel_problem_lstday(self):
        requests = [(log['time'].strftime('%Y%m%d'), log['event']) for log in self.logs if log['time'].strftime('%Y%m%d') == self.end]
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
        self.feature_keys.append('module_and_problem_count_lstday')
        self.feature_values.append(cnt)

        self.feature_keys.append('module_problem_ratio_lstday')
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

        self.feature_keys.append('module_not_problem_count_lstday')
        self.feature_values.append(cnt)

        self.feature_keys.append('module_not_problem_ratio_lstday')
        self.feature_values.append(ratio)
        return self
