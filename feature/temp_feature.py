#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math

from data import *
from util import *

class TempFeatureFactory:
    def __init__(self, features = [], modules=None):
        self.features = features
        self.modules = modules

    def set_features(self, features):
        self.features = features

    def append_features(self, features):
        self.features += features


    def gen_features(self, dataset):
        for timeline in dataset.iter_timelines():
            eid = timeline.eid

            values = timeline_features(timeline, self.features, self.modules)
            target = dataset.get_label(eid)
            yield (eid, target, values)

    def dump(self, dataset, filename):
        output = open(filename, 'w')
        if dataset.isTest():
            output.write("eid,cid,%s\n" % ','.join(self.features))
        else:
            output.write("eid,cid,target,%s\n" % ','.join(self.features))

        for eid, target, values in self.gen_features(dataset):
            if not dataset.isTest():
                output.write('%s,%s,%s,%s\n' % (eid, dataset.course_by_eid(eid), target, (','.join(['%.3f' % v for v in values]))))
            else:
                output.write('%s,%s,%s\n' % (eid, dataset.course_by_eid(eid), (','.join(['%.3f' % v for v in values]))))
        output.close()

def timeline_features(timeline, features, modules=None):
    # ['problem', 'wiki', 'access', 'nagivate', 'discussion', 'video']
    # 2 - 7
    values = []

    d = timeline.duration("d")
    if '#duration' in features:
        values.append(d)

    # total number of requests, e.g., access and video event
    browser_et_times = timeline.event_times("browser")
    if "#browser_request" in features:
        x = browser_et_times["access"]
        x += browser_et_times['video']
        x += browser_et_times['nagivate']
        x += browser_et_times['wiki']
        x += browser_et_times['problem']
        x += browser_et_times['discussion']

        values.append(x)


    if '#browser_access' in features:
        x = browser_et_times["access"]
        values.append(x)

    if '#browser_video' in features:
        x = browser_et_times["video"]
        values.append(x)

    # if '#browser_discussion' in features:
    #     x = browser_et_times["discussion"]
    #     values.append(x)
    #
    # if '#browser_wiki' in features:
    #     x = browser_et_times['wiki']
    #     values.append(x)

    if "#browser_problem" in features:
        x = browser_et_times['problem']
        values.append(x)

    # server side
    server_et_times = timeline.event_times("server")
    if "#server_request" in features:
        x = server_et_times["access"]
        x += server_et_times['video']
        x += server_et_times['nagivate']
        x += server_et_times['wiki']
        x += server_et_times['problem']
        x += server_et_times['discussion']

        values.append(x)


    if '#server_access' in features:
        x = server_et_times["access"]
        values.append(x)

    # if '#server_video' in features:
    #     x = server_et_times["video"]
    #     values.append(x)

    if '#server_discussion' in features:
        x = server_et_times["discussion"]
        values.append(x)

    if '#server_wiki' in features:
        x = server_et_times['wiki']
        values.append(x)

    if "#server_problem" in features:
        x = server_et_times['problem']
        values.append(x)



    # Number of active days
    if '#active_days' in features:
        x = timeline.active_days()
        values.append(x)

    # Number of sessions
    sessions = timeline.split_sessions()
    num_session = len(sessions)
    if '#session' in features:
        x = num_session
        values.append(x)

    if '#avg_requests_per_session' in features:
        x = 0.0
        for s in sessions:
            x += len(s)
        x = x / (num_session + 0.01)
        values.append(x)


    # average number of video click actions per session
    et_times = timeline.event_times()
    if '#avg_video_per_session' in features:
        x = et_times['video'] / (num_session + 0.0)
        values.append(x)

    if '#avg_discuss_per_session' in features:
        x = et_times['discussion'] / (num_session + 0.0)
        values.append(x)

    if '#avg_access_per_session' in features:
        x = et_times['access'] / (num_session + 0.0)
        values.append(x)

    if '#avg_nagivate_per_session' in features:
        x = et_times['nagivate'] / (num_session + 0.0)
        values.append(x)

    if '#avg_problem_per_session' in features:
        x = et_times['problem'] / (num_session + 0.0)
        values.append(x)

    # Most common request time:
    # night time from 19:00 to 6:59 in the morning and the other half day as day time

    day_times = 0
    night_times = 0

    weekend_times = 0.0
    weekday_times = 0.0

    for e in timeline.events:
        etype = e.get_event()
        if etype not in ['problem', 'access', 'wiki', 'video', 'discussion']:
            continue

        w = get_weekday(e.get_time())

        if w in ["Sunday", "Saturday"]:
            weekend_times += 1.0
        else:
            weekday_times += 1.0

        d, t = e.get_time().split('T')
        h, m, s = t.split(':')
        h = int(h)
        if h < 19 and h >= 7:
            day_times += 1.0
        else:
            night_times += 1

    if '#daytime_ratio' in features:
        if (day_times + night_times) > 0:
            x = day_times / (day_times + night_times + 0.0)
        else:
            x = 0.0
        values.append(x)

    if '#night_time' in features and '#day_time' in features:

        if night_times >= day_times:
            values.append(1)
            values.append(0)
        else:
            values.append(0)
            values.append(1)


    # most common request week day
    if '#weekend_ratio' in features:

        if (weekend_times + weekday_times) > 0:
            x = weekend_times / (weekend_times + weekday_times + 0.0)
        else:
            x = 0.0
        values.append(x)

    if '#weekend_time' in features and '#weekday_time' in features:
        if weekend_times >= weekday_times:
            values.append(1.0)
            values.append(0.0)
        else:
            values.append(0.0)
            values.append(1.0)


    days = [d["date"] for d in timeline.events_by_days]
    days.sort()

    last_day = get_timestamp(days[-1], format='%Y-%m-%d')
    start_day = last_day - 7 * 24 * 60 * 60
    for i, d in enumerate(days):
        t = get_timestamp(d, '%Y-%m-%d')
        if t >= start_day:
            break

    if '#active_days_last_week' in features:
        values.append(len(days) - i)

    start_day_2w = last_day - 14 * 24 * 60 * 60
    last_week_event_times = {}
    last_2week_event_times = {}
    for etype in EVENT_TYPES:
        last_week_event_times[etype] = 0.0
        last_2week_event_times[etype] = 0.0

    for e in timeline.events:
        stm = e.get_stamp()
        if stm >= start_day:
            etype = e.get_event()
            if etype in EVENT_TYPES:
                last_week_event_times[etype] += 1.0
        if stm >= start_day_2w:
            etype = e.get_event()
            if etype in EVENT_TYPES:
                last_2week_event_times[etype] += 1.0


    if '#request_last_week' in features:
        x = last_week_event_times["access"]
        x += last_week_event_times['video']
        x += last_week_event_times['nagivate']
        x += last_week_event_times['wiki']
        x += last_week_event_times['problem']
        x += last_week_event_times['discussion']
        values.append(x)

    if '#access_last_week' in features:
        x = last_week_event_times['access']
        values.append(x)

    if '#video_last_week' in features:
        x = last_week_event_times['video']
        values.append(x)

    if '#problem_last_week' in features:
        x = last_week_event_times['problem']
        values.append(x)

    sessions_last_week = timeline.split_sessions(start_day)
    num_session = len(sessions_last_week)

    if '#session_last_week' in features:
        values.append(num_session)

    if '#avg_requests_per_session_last_week' in features:
        x = 0
        for s in sessions_last_week:
            x += len(s)

        x = x / (num_session + 0.01)
        values.append(x)

    if '#avg_access_per_session_last_week' in features:
        x = last_week_event_times['access'] / (num_session + 0.01)
        values.append(x)

    if '#avg_video_per_session_last_week' in features:
        x = last_week_event_times['video'] / (num_session + 0.01)
        values.append(x)

    if '#avg_problem_per_session_last_week' in features:
        x = last_week_event_times['problem'] / (num_session + 0.01)
        values.append(x)

    if '#request_last_2week' in features:
        x = last_2week_event_times["access"]
        x += last_2week_event_times['video']
        x += last_2week_event_times['nagivate']
        x += last_2week_event_times['wiki']
        x += last_2week_event_times['problem']
        x += last_2week_event_times['discussion']
        values.append(x)

    if '#access_last_2week' in features:
        x = last_2week_event_times['access']
        values.append(x)

    if '#video_last_2week' in features:
        x = last_2week_event_times['video']
        values.append(x)

    if '#problem_last_2week' in features:
        x = last_2week_event_times['problem']
        values.append(x)

    #session_last_2week
    sessions_last_2week = timeline.split_sessions(start_day_2w)
    num_session = len(sessions_last_2week)
    if '#session_last_2week' in features:
        values.append(num_session)

    if "#avg_requests_per_session_last_2week" in features:
        x = 0.0
        for s in sessions_last_2week:
            x += len(s)
        x = x / (num_session + 0.01)
        values.append(x)

    if "#avg_access_per_session_last_2week" in features:
        x = last_2week_event_times['access'] / (num_session + 0.01)
        values.append(x)

    if "#avg_video_per_session_last_2week" in features:
        x = last_2week_event_times['video'] / (num_session + 0.01)
        values.append(x)


    if "#avg_problem_per_session_last_2week" in features:
        x = last_2week_event_times['problem'] / (num_session + 0.01)
        values.append(x)

    times_hours = {}
    for h in range(24):
        times_hours[h] = 0.0

    total = 0.0
    for e in timeline.events:
        d, t = e.get_time().split('T')
        h = int(t.split(':')[0])
        if stm >= start_day_2w:
            etype = e.get_event()
            if etype in EVENT_TYPES:
                times_hours[h] += 1.0
                total += 1.0

    if '#0_6h_request':
        x = sum(map(lambda x: times_hours[x], range(7)))
        if total > 0:
            x = x / total
        values.append(x)

    if '#6-9h_request':
        x = sum(map(lambda x: times_hours[x], range(6, 10)))
        if total > 0:
            x = x / total
        values.append(x)

    if '#8-12h_request':
        x = sum(map(lambda x: times_hours[x], range(8, 13)))
        if total > 0:
            x = x / total
        values.append(x)

    if '#12-18h_request':
        x = sum(map(lambda x: times_hours[x], range(12, 18)))
        if total > 0:
            x = x / total
        values.append(x)

    if '#17-20h_request':
        x = sum(map(lambda x: times_hours[x], range(17, 20)))
        if total > 0:
            x = x / total
        values.append(x)

    if '#19-24h_request':
        x = sum(map(lambda x: times_hours[x], range(19, 24)))
        if total > 0:
            x = x / total
        values.append(x)


    lags = []
    start = days[0]
    sum_lags = 0.0
    for i in range(1, len(days)):
        end = days[i]
        lag = get_duration(start, end, format='%Y-%m-%d') / (60 * 60 * 24)
        sum_lags += lag
        lags.append(lag)
        start = end
    if '#total_lagging' in features:
        values.append(sum_lags)

    if '#lagging_3days_times' in features:
        x = 0
        for l in lags:
            if l >= 3:
                x += 1.0
        values.append(x)

    if '#lagging_1week_times' in features:
        x = 0
        for l in lags:
            if lag >= 6:
                x += 1.0
        values.append(x)

    if '#lagging_2week_times' in features:
        x = 0
        for l in lags:
            if lags >= 12:
                x += 1.0
        values.append(x)

    if '#avg_lagging' in features:
        if len(lags) > 0:
            avg_lag = sum_lags / len(lags)
        else:
            avg_lag = 0.0
        values.append(avg_lag)

    if '#std_lagging' in features:
        std = 0.0
        for lag in lags:
            std += (lag - avg_lag) * (lag - avg_lag)
        if len(lags) == 0:
            std = 0
        else:
            std = math.sqrt(std / len(lags))
        values.append(std)

    if '#lagging2week' in features and '#lagging<2week' in features:
        if sum_lags >= 14:
            values.append(1.0)
            values.append(0.0)
        else:
            values.append(0.0)
            values.append(1.0)


    # # problem lagging
    # x_2d = 0
    # x_1w = 0
    # x_2w = 0
    # for e in timeline.events:
    #     et = e.get_event()
    #     if et != 'problem':
    #         continue
    #
    #     mid = e.get_obj()
    #
    #     m = modules.get_module_by_mid(mid)
    #     if m:
    #         rt = m.get_start()
    #         d = get_duration(rt, e.get_stamp())
    #         if d >= 2 * (60 * 60 * 24):
    #             x_2d += 1.0
    #         if d >= 7 * (60 * 60 * 24):
    #             x_1w += 1.0
    #         if d >= 14 * (60 * 60 * 24):
    #             x_2w += 1.0
    #
    # if '#problem_2day_lagging' in features:
    #     values.append(x_2d)
    # if '#problem_1week_lagging' in features:
    #     values.append(x_1w)
    # if '#problem_2week_lagging' in features:
    #     values.append(x_2w)

    # # video lagging
    # x_2d = 0
    # x_1w = 0
    # x_2w = 0
    # for e in timeline.events:
    #     et = e.get_event()
    #     if et != 'video':
    #         continue
    #
    #     mid = e.get_obj()
    #
    #     m = modules.get_module_by_mid(mid)
    #     if m:
    #         rt = m.get_start()
    #         d = get_duration(rt, e.get_stamp)
    #         if d >= 2 * (60 * 60 * 24):
    #             x_2d += 1.0
    #         if d >= 7 * (60 * 60 * 24):
    #             x_1w += 1.0
    #         if d >= 14 * (60 * 60 * 24):
    #             x_2w += 1.0
    # if '#video_2day_lagging' in features:
    #     values.append(x_2d)
    # if "#video_1week_lagging" in features:
    #     values.append(x_1w)
    # if '#video_2week_lagging' in features:
    #     values.append(x_2w)

    # access
    x_2d = 0
    x_1w = 0
    x_2w = 0
    for e in timeline.events:
        et = e.get_event()
        if et != 'access':
            continue

        mid = e.get_obj()

        m = modules.get_module_by_mid(mid)
        if m:
            rt = modules.get_start(mid)
            if rt is None:
                continue
            # print rt,
            # print ' ',
            # print e.get_stamp()
            d = e.get_stamp() - rt
            if d >= 2 * (60 * 60 * 24):
                x_2d += 1.0
            if d >= 7 * (60 * 60 * 24):
                x_1w += 1.0
            if d >= 14 * (60 * 60 * 24):
                x_2w += 1.0

    if '#access_2day_lagging' in features:
        values.append(x_2d)
    if "#access_1week_lagging" in features:
        values.append(x_1w)
    if '#access_2week_lagging' in features:
        values.append(x_2w)


    # browser vs server
    if '#ratio_browser' in features:
        x = 0
        s = 0
        for e in timeline.events:
            es = e.get_source()
            if 'browser' == es:
                x += 1.0
            s += 1.0

        values.append(x / s)

    return values
