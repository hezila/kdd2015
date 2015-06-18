#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import Counter
import datetime
import os
import math, random


from feature_bag import FeatureBag

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-16-2015'

base_dir = os.path.dirname(__file__)
event_types = ['problem', 'video', 'access', 'wiki', 'discussion', 'nagivate', 'page_close']
server_events = ['access', 'wiki', 'discussion', 'problem']
browser_events = ['access', 'video', 'problem', 'page_close']

class WeekFeatureBag(FeatureBag):
    def __init__(self, enrollment_id, label, week, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)
        self.logs = sorted(logs, key=lambda x: x['time'])
        self.start = self.logs[0]['time'].strftime('%Y%m%d')
        self.start_day = datetime.datetime.strptime(self.start, '%Y%m%d')
        self.end = self.logs[-1]['time'].strftime('%Y%m%d')
        self.end_day = datetime.datetime.strptime(self.end, '%Y%m%d')
        self.label = label
        if week:
            self.week = week
        else:
            self.week = int((self.end_day - self.start_day).days / 7)

    def gen_sub_bags(self):

        start_weekday = self.start_day.weekday()
        next_monday = self.start_day + datetime.timedelta(days=8 - start_weekday)

        w = 1
        for i in range(len(self.logs)):
            log = self.logs[i]
            if log['time'] > next_monday:
                bag_logs = self.logs[:i]
                next_monday = next_monday + datetime.timedelta(days=7)
                lag = (log['time'] - self.logs[i-1]['time']).days
                if lag > 9:
                    l = 1
                else:
                    l = 0
                sub_bag = WeekFeatureBag('{0}-{1}w'.format(self.enrollment_id, w), l, w, bag_logs, [], [])
                w += 1
                

                if self.label == 1 and random.random() > 0.5:
                    print '{0}-{1}w-{2}'.format(self.enrollment_id, w, l)
                    yield sub_bag

        # w = (self.end_day - self.start_day).days + 1
        # w = int(w / 7)
        # if w == 0:
        #     w = 1

        print '{0}-{1}w-{2}'.format(self.enrollment_id, w, self.label)
        yield WeekFeatureBag('{0}-{1}w'.format(self.enrollment_id, w), self.label, w, self.logs, [], [])


    def extract_duration_days(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        duration = (request_dates[-1] - request_dates[0]).days + 1.0

        self.feature_keys.append('duration')
        self.feature_values.append(duration + 0.0)

        return self

    def extract_request_count(self):
        self.feature_keys.append("request_count")
        x = len(self.logs) + 0.0
        self.feature_values.append(x)

        return self

    def extract_active_days(self):
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs])
        d = len(request_dates) + 0.0

        self.feature_keys.append('active_days')
        self.feature_values.append(d)

        return self

    def extract_avg_active_days(self):
        request_dates = set([log['time'].strftime('%Y%m%d') for log in self.logs])
        active_days = len(request_dates)
        request_dates = sorted(list(request_dates))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        duration = (request_dates[-1] - request_dates[0]).days + 1.0

        weeks = int(duration / 7.0)
        if weeks == 0:
            weeks = 1
        avg = active_days / (weeks + 0.0)
        self.feature_keys.append('avg_active_days')
        self.feature_values.append(avg)
        return self




    def extract_request_lag_min(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            lags = [10] # one day in this case
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        self.feature_keys.append('min_request_lag')
        self.feature_values.append(min(lags) + 0.0)
        return self

    def extract_request_lag_max(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            lags = [10]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        self.feature_keys.append('max_request_lag')
        self.feature_values.append(max(lags) + 0.0)
        return self


    def extract_request_lag_mean(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            lags = [10]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
        self.feature_keys.append('mean_request_lag')
        self.feature_values.append(sum(lags) / (len(lags) + 0.0)) # avoid numpy to use pypy
        return self

    def extract_request_lag_var(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]
        if len(request_dates) == 1:
            var = 3.0
        elif len(request_dates) == 2:
            var = (request_dates[1] - request_dates[0]).days + 1.0
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]
            mean = sum(lags) / (len(lags) + 0.0)
            var = 1.0 / (len(lags) - 1) * sum([(l - mean)**2 for l in lags])
        self.feature_keys.append('var_request_lag')
        self.feature_values.append(var)
        return self

    def extract_request_lag_3days(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]

        if len(request_dates) == 1:
            lags = [1]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]

        cnt = 0.0
        for l in lags:
            if l >= 3:
                cnt += 1.0

        self.feature_keys.append('request_lag_3days')
        self.feature_values.append(cnt)
        return self

    def extract_request_lag_3days_ratio(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]

        if len(request_dates) == 1:
            lags = [1]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]

        cnt = 0.0
        for l in lags:
            if l >= 3:
                cnt += 1.0
        ratio = cnt / len(lags)
        self.feature_keys.append('request_lag_3days_ratio')
        self.feature_values.append(ratio)
        return self

    def extract_request_lag_5days(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]

        if len(request_dates) == 1:
            lags = [1]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]

        cnt = 0.0
        for l in lags:
            if l >= 5:
                cnt += 1.0

        self.feature_keys.append('request_lag_5days')
        self.feature_values.append(cnt)
        return self

    def extract_request_lag_5days_ratio(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]

        if len(request_dates) == 1:
            lags = [1]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]

        cnt = 0.0
        for l in lags:
            if l >= 5:
                cnt += 1.0

        ratio = cnt / len(lags)

        self.feature_keys.append('request_lag_5days_ratio')
        self.feature_values.append(ratio)
        return self

    def extract_request_lag_1week(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]

        if len(request_dates) == 1:
            lags = [7]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]

        cnt = 0.0
        for l in lags:
            if l >= 7:
                cnt += 1.0

        self.feature_keys.append('request_lag_1week')
        self.feature_values.append(cnt)
        return self

    def extract_request_lag_1week_ratio(self):
        request_dates = sorted(list(set([log['time'].strftime('%Y%m%d') for log in self.logs])))
        request_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in request_dates]

        if len(request_dates) == 1:
            lags = [1]
        else:
            lags = [(request_dates[i+1] - request_dates[i]).days for i in range(len(request_dates) - 1)]

        cnt = 0.0
        for l in lags:
            if l >= 7:
                cnt += 1.0

        ratio = cnt / len(lags)

        self.feature_keys.append('request_lag_1week_ratio')
        self.feature_values.append(ratio)
        return self


    def extract_request_hours(self):
        request_hours = sorted([log['time'].strftime('%H') for log in self.logs])
        counter = Counter(request_hours)

        # 0am - 6am
        cnt = 0.0
        for i in range(7):
            h = '{0:02d}'.format(i)
            if h in counter:
                cnt = counter[h]
        self.feature_keys.append("request_hour_0_6am")
        self.feature_values.append(cnt / len(request_hours))

        # 6am - 9am
        cnt = 0.0
        for i in range(6, 10):
            h = '{0:02d}'.format(i)
            if h in counter:
                cnt = counter[h]
        self.feature_keys.append('request_hour_6_9am')
        self.feature_values.append(cnt / len(request_hours))

        # 8-12am
        cnt = 0.0
        for i in range(8, 13):
            h = '{0:02d}'.format(i)
            if h in counter:
                cnt = counter[h]
        self.feature_keys.append('request_hour_8_12am')
        self.feature_values.append(cnt / len(request_hours))

        # 12 - 18pm
        cnt = 0.0
        for i in range(12, 19):
            h = '{0:02d}'.format(i)
            if h in counter:
                cnt = counter[h]
        self.feature_keys.append('request_hour_12_18pm')
        self.feature_values.append(cnt / len(request_hours))

        # 17 - 20pm
        cnt = 0.0
        for i in range(17, 21):
            h = '{0:02d}'.format(i)
            if h in counter:
                cnt = counter[h]
        self.feature_keys.append('request_hour_17_20pm')
        self.feature_values.append(cnt / len(request_hours))

        # 19 - 24pm
        cnt = 0.0
        for i in range(19, 25):
            h = '{0:02d}'.format(i)
            if h in counter:
                cnt = counter[h]
        self.feature_keys.append('request_hour_19_24pm')
        self.feature_values.append(cnt / len(request_hours))


        probs = []
        for h in range(24):
            if h in counter:
                probs.append(counter[h] / len(request_hours))
            # else:
            #     probs.append(0.0)



        ent = 0.

        # Compute standard entropy.
        for i in probs:

            ent -= i * math.log(i)

        self.feature_keys.append('request_hour_entropy')
        self.feature_values.append(ent)
        # for i in xrange(24):
        #     h = '{0:02d}'.format(i)
        #     cnt = 0
        #     if h in counter:
        #         cnt = counter[h]
        #     self.feature_keys.append('request_hour_{0}'.format(h))
        #     self.feature_values.append(cnt + 0.0)
        return self



    def extract_request_hour_count(self):
        request_hours = set([int(log['time'].strftime('%H')) for log in self.logs])
        self.feature_keys.append('request_hour_count')
        self.feature_values.append(len(request_hours) + 0.0)
        return self

    def extract_request_hour_mean(self):
        request_hours = [int(log['time'].strftime('%H')) for log in self.logs]
        self.feature_keys.append('request_hour_mean')

        self.feature_values.append(sum(request_hours) / (len(request_hours) + 0.0))
        return self

    def extract_request_hour_var(self):
        request_hours = [int(log['time'].strftime('%H')) for log in self.logs]
        n = len(request_hours) + 0.0
        mean = sum(request_hours) / n

        var = 1.0 / (n - 1) * sum([(h - mean)**2 for h in request_hours]) if n > 1 else 6

        self.feature_keys.append('request_hour_var')
        self.feature_values.append(var)
        return self


    def extract_request_weekend_count(self):
        ws = [log['time'].weekday() for log in self.logs]
        d = len([1 for w in ws if w > 5]) + 0.0
        self.feature_keys.append('request_weekend_count')
        self.feature_values.append(d)

        if d > (len(ws) - d):
            self.feature_keys.append('more_weekend')
            self.feature_values.append(1.0)

            self.feature_keys.append('more_weekday')
            self.feature_values.append(0.0)
        else:
            self.feature_keys.append('more_weekend')
            self.feature_values.append(0.0)

            self.feature_keys.append('more_weekday')
            self.feature_values.append(1.0)
        return self


    def extract_request_weekend_percentage(self):
        ws = [log['time'].weekday() for log in self.logs]
        self.feature_keys.append('request_weekend_percentage')
        self.feature_values.append(float(sum([1 for w in ws if w > 5])) / len(ws))
        return self


    def extract_session_count(self):
        return self

    def extract_session_mean(self):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])
        sessions = []
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 30:
                    cnt += 1.0
            sessions.append(cnt + 1.0)
        self.feature_keys.append('session_mean')
        self.feature_values.append(sum(sessions) / (len(sessions) + 0.0))

        return self

    def extract_session_var(self):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])
        sessions = []
        for k, v in daily_timeline.items():
            cnt = 0
            for i in range(len(v) - 1):
                if (v[i+1] - v[i]).seconds >= 60 * 30:
                    cnt += 1.0
            sessions.append(cnt + 1.0)

        mean = sum(sessions) / (len(sessions) + 0.0)
        if len(sessions) == 1:
            var = 0.5
        elif len(sessions) == 2:
            var = (sessions[1] - sessions[0])**2
        else:
            var = 1.0 / (len(sessions) - 1.0) * sum([(x - mean)**2 for x in sessions])
        self.feature_keys.append('session_var')
        self.feature_values.append(var)
        return self

    def extract_staytime_min(self):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        stay_times = []
        start = None
        for k, v in daily_timeline.items():
            start = v[0]
            stay = 0.0
            for i in range(len(v) - 1):
                l = (v[i+1] - v[i]).seconds
                if l >= 60*30:
                    stay += (v[i] - start).seconds
                    start = v[i+1]
            stay += (v[-1] - start).seconds
            stay_times.append(stay / 60.0)

        self.feature_keys.append('staytime_min')
        self.feature_values.append(min(stay_times))
        return self

    def extract_staytime_max(self):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        stay_times = []
        start = None
        for k, v in daily_timeline.items():
            start = v[0]
            stay = 0.0
            for i in range(len(v) - 1):
                l = (v[i+1] - v[i]).seconds
                if l >= 60*30:
                    stay += (v[i] - start).seconds
                    start = v[i+1]
            stay += (v[-1] - start).seconds
            stay_times.append(stay / 60.0)

        self.feature_keys.append('staytime_max')
        self.feature_values.append(max(stay_times))
        return self

    def extract_staytime_mean(self):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        stay_times = []
        start = None
        for k, v in daily_timeline.items():
            start = v[0]
            stay = 0.0
            for i in range(len(v) - 1):
                l = (v[i+1] - v[i]).seconds
                if l >= 60*30:
                    stay += (v[i] - start).seconds
                    start = v[i+1]
            stay += (v[-1] - start).seconds
            stay_times.append(stay / 60.0)

        self.feature_keys.append('staytime_mean')
        self.feature_values.append(sum(stay_times) / (len(stay_times) + 0.0))
        return self

    def extract_staytime_var(self):
        request_dates = [(log['time'].strftime('%Y%m%d'), log['time']) for log in self.logs]
        request_dates = sorted(request_dates, key=lambda x: x[1])
        daily_timeline = {}
        for d in request_dates:
            daily_timeline.setdefault(d[0], []).append(d[1])

        stay_times = []
        start = None
        for k, v in daily_timeline.items():
            start = v[0]
            stay = 0.0
            for i in range(len(v) - 1):
                l = (v[i+1] - v[i]).seconds
                if l >= 60*30:
                    stay += (v[i] - start).seconds
                    start = v[i+1]
            stay += (v[-1] - start).seconds
            stay_times.append(stay / 60.0)

        mean = sum(stay_times) / (len(stay_times) + 0.0)
        if len(stay_times) == 1:
            var = math.sqrt(stay_times[0])
        else:
            var = 1.0 / (len(stay_times) - 1.0) * sum((v - mean)**2 for v in stay_times)
        self.feature_keys.append('staytime_var')
        self.feature_values.append(var)

        return self

    def extract_server_event_count(self):

        events = sorted([log['event'] for log in self.logs if log['source'] == 'server'])
        counter = Counter(events)
        for event_type in server_events:
            cnt = 0
            if event_type in counter:
                cnt = counter[event_type]
            self.feature_keys.append('event_server_{0}_count'.format(event_type))
            self.feature_values.append(cnt)
        return self

    def extract_browser_event_count(self):

        events = sorted([log['event'] for log in self.logs if log['source'] == 'browser'])
        counter = Counter(events)
        for event_type in browser_events:
            cnt = 0
            if event_type in counter:
                cnt = counter[event_type]
            self.feature_keys.append('event_browser_{0}_count'.format(event_type))
            self.feature_values.append(cnt)

        return self

    def extract_event_count(self):
        events = sorted([log['event'] for log in self.logs if log['event'] in ['access', 'problem']])
        counter = Counter(events)
        for event_type in ['access', 'problem']:
            cnt = 0
            if event_type in counter:
                cnt = counter[event_type]
            self.feature_keys.append('event_{0}_count'.format(event_type))
            self.feature_values.append(cnt)
        return self

    def extract_event_percentage(self):
        events = sorted([log['event'] for log in self.logs])
        counter = Counter(events)
        for event_type in event_types:
            cnt = 0
            if event_type in counter:
                cnt = counter[event_type]
            self.feature_keys.append('event_{0}_percentage'.format(event_type))
            self.feature_values.append(float(cnt) / len(events))
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
                night_times += 1


        self.feature_keys.append('daytime_ratio')
        self.feature_values.append(day_times / (day_times + night_times))

        if night_times >= day_times:
            self.feature_keys.append('day_time')
            self.feature_values.append(0.0)

            self.feature_keys.append('night_time')
            self.feature_values.append(1.0)

        else:
            self.feature_keys.append('day_time')
            self.feature_values.append(1.0)

            self.feature_keys.append('night_time')
            self.feature_values.append(0.0)
        return self

    def extract_video_over10minutes_count(self):
        cnt = 0
        for i in xrange(len(self.logs) - 1):
            if self.logs[i]['event'] != 'video':
                continue

            time_delta = self.logs[i+1]['time'] - self.logs[i]['time']

            if 600 <= time_delta.seconds <= 18000 and time_delta.days == 0:
                cnt += 1.0
        self.feature_keys.append('video_over10minutes_count')
        self.feature_values.append(cnt)
        return self


    def extract_video_over10minutes_count_lst1week(self):
        last_day = self.logs[-1]['time'].strftime('%Y%m%d')
        last_day = datetime.datetime.strptime(last_day, '%Y%m%d')
        start_date = last_day - datetime.timedelta(days=7)
        cnt = 0
        for i in xrange(len(self.logs) - 1):
            if self.logs[i]['event'] != 'video':
                continue
            if self.logs[i]['time'] < start_date:
                continue

            time_delta = self.logs[i+1]['time'] - self.logs[i]['time']

            if 600 <= time_delta.seconds <= 18000 and time_delta.days == 0:
                cnt += 1.0
        self.feature_keys.append('video_over10minutes_count_lst1week')
        self.feature_values.append(cnt)
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


    def extract_problem_over3minutes_count_lst1week(self):
        last_day = self.logs[-1]['time'].strftime('%Y%m%d')
        last_day = datetime.datetime.strptime(last_day, '%Y%m%d')
        start_date = last_day - datetime.timedelta(days=7)

        cnt = 0
        for i in xrange(len(self.logs) - 1):
            if self.logs[i]['event'] != 'problem':
                continue

            if self.logs[i]['time'] < start_date:
                continue


            time_delta = self.logs[i+1]['time'] - self.logs[i]['time']

            if 180 <= time_delta.seconds <= 18000 and time_delta.days == 0:
                cnt += 1.0
        self.feature_keys.append('problem_over3minutes_count_lst1week')
        self.feature_values.append(cnt)
        return self


    def extract_request_count_lst1weeks(self):
        request_dates = sorted([log['time'] for log in self.logs])
        start_date = request_dates[-1] - datetime.timedelta(days=7)
        cnt = 0
        for d in request_dates:
            if d >= start_date:
                cnt += 1.0

        self.feature_keys.append('request_count_lst1weeks')
        self.feature_values.append(cnt)

        return self




    def extract_server_event_count_lst1weeks(self):
        request_dates = sorted([log['time'] for log in self.logs])
        start_date = request_dates[-1] - datetime.timedelta(days=7)
        event_counts = {}
        for log in self.logs:
            if log['source'] != 'server':
                continue

            if log['time'] >= start_date:
                event_type = log['event']
                if event_type not in event_counts:
                    event_counts[event_type] = 1.0
                else:
                    event_counts[event_type] += 1.0
        for event_type in server_events:
            cnt = 0
            if event_type in event_counts:
                cnt = event_counts[event_type]
            self.feature_keys.append('{0}_event_server_count_lst1weeks'.format(event_type))
            self.feature_values.append(cnt)
        return self



    def extract_browser_event_count_lst1weeks(self):
        request_dates = sorted([log['time'] for log in self.logs])
        start_date = request_dates[-1] - datetime.timedelta(days=7)
        event_counts = {}
        for log in self.logs:
            if log['source'] != 'browser':
                continue
            if log['time'] >= start_date:
                event_type = log['event']
                if event_type not in event_counts:
                    event_counts[event_type] = 1.0
                else:
                    event_counts[event_type] += 1.0
        for event_type in browser_events:
            cnt = 0
            if event_type in event_counts:
                cnt = event_counts[event_type]
            self.feature_keys.append('{0}_event_browser_count_lst1weeks'.format(event_type))
            self.feature_values.append(cnt)
        return self



    def extract_activedays_lst1weeks(self):
        request_dates = sorted([log['time'] for log in self.logs])
        start_date = request_dates[-1] - datetime.timedelta(days=7)
        days = set([log['time'].strftime('%Y%m%d') for log in self.logs if log['time'] >= start_date])
        self.feature_keys.append('activedays_lst1weeks')
        self.feature_values.append(len(days))
        return self



    def extract_month_fst_access(self):
        request_dates = sorted([log['time'] for log in self.logs])
        m = int(request_dates[0].strftime('%m'))
        for i in range(1, 13):
            self.feature_keys.append('fst_access_month_{0}'.format(i))

            if m == i:
                self.feature_values.append(1.0)
            else:
                self.feature_values.append(0.0)

        return self

    def extract_month_lst_access(self):
        request_dates = sorted([log['time'] for log in self.logs])
        m = int(request_dates[-1].strftime('%m'))
        for i in range(1, 13):
            self.feature_keys.append('lst_access_month_{0}'.format(i))

            if m == i:
                self.feature_values.append(1.0)
            else:
                self.feature_values.append(0.0)
        return self

    def extract_event_days_per_week(self):
        start_date = datetime.datetime(2014, 5, 13)
        event_week = {}
        for event_type in event_types:
            event_week[event_type] = [0 for i in range(82/7+1)]
        targets = set(['{0},{1}'.format(log['time'].strftime('%Y%m%d'), log['event']) for log in self.logs])
        for target in targets:
            d, e = target.split(',')
            diff = (datetime.datetime.strptime(d, '%Y%m%d')-start_date).days/7
            if 0 <= diff < len(event_week[e]):
                event_week[e][diff] += 1
        for event, weeks in event_week.items():
            for i, week in enumerate(weeks):
                self.feature_keys.append('event_days_{0}_week{1:02d}'.format(event, i))
                self.feature_values.append(week)
        return self

    def extract_video_over10minutes_count_per_week(self):
        start_date = datetime.datetime(2014, 5, 13)
        weeks = [0 for i in range(82/7+1)]
        for i in range(len(self.logs)-1):
            if self.logs[i]['event'] != 'video':
                continue
            time_delta = self.logs[i+1]['time']-self.logs[i]['time']
            if 600 < time_delta.seconds < 18000 and time_delta.days == 0:
                diff = (self.logs[i+1]['time']-start_date).days/7
                if 0 <= diff < len(weeks):
                    weeks[diff] += 1
        for i, week in enumerate(weeks):
            self.feature_keys.append('video_over10minutes_week{0:02d}'.format(i))
            self.feature_values.append(week)
        return self

    def extract_problem_over3minutes_count_per_week(self):
        start_date = datetime.datetime(2014, 5, 13)
        weeks = [0 for i in range(82/7+1)]
        for i in range(len(self.logs)-1):
            if self.logs[i]['event'] != 'video':
                continue
            time_delta = self.logs[i+1]['time']-self.logs[i]['time']
            if 180 < time_delta.seconds < 18000 and time_delta.days == 0:
                diff = (self.logs[i+1]['time']-start_date).days/7
                if 0 <= diff < len(weeks):
                    weeks[diff] += 1
        for i, week in enumerate(weeks):
            self.feature_keys.append('problem_over3minutes_week{0:02d}'.format(i))
            self.feature_values.append(week)
        return self

    def extract_source_count(self):
        sources = [log['source'] for log in self.logs]
        server_cnt = len([source for source in sources if source == 'server'])
        browser_cnt = len(sources) - server_cnt
        self.feature_keys.append('source_server_count')
        self.feature_values.append(server_cnt)

        self.feature_keys.append('source_browser_count')
        self.feature_values.append(browser_cnt)

        self.feature_keys.append('browser_ratio')
        self.feature_values.append(browser_cnt / (len(sources) + 0.0))
        return self
