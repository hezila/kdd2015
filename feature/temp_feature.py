#!/usr/bin/env python
#-*- coding: utf-8 -*-

from data import *
from util import *

class TempFeatureFactory:
    def __init__(self, features = []):
        self.features = features

    def set_features(self, features):
        self.features = features

    def append_features(self, features):
        self.features += features


    def gen_features(self, dataset):
        for timeline in dataset.iter_timelines():
            eid = timeline.eid

            values = timeline_features(timeline, self.features)
            target = dataset.get_label(eid)
            yield (eid, target, values)

    def dump(self, dataset, filename):
        output = open(filename, 'w')
        if dataset.isTest():
            output.write("eid,%s\n" % ','.join(self.features))
        else:
            output.write("eid,target,%s\n" % ','.join(self.features))

        for eid, target, values in self.gen_features(dataset):
            if not dataset.isTest():
                output.write('%s,%s,%s,%s\n' % (eid, dataset.course_by_eid(eid), target, (','.join(['%.3f' % v for v in values]))))
            else:
                output.write('%s,%s,%s\n' % (eid, dataset.course_by_eid(eid), (','.join(['%.3f' % v for v in values]))))
        output.close()

def timeline_features(timeline, features):
    # ['problem', 'wiki', 'access', 'nagivate', 'discussion', 'video']
    # 2 - 7
    values = []

    d = timeline.duration("d")
    if '#duration' in features:
        values.append(d)

    # total number of requests, e.g., access and video event
    et_times = timeline.event_times()
    if "#request" in features:
        x = et_times["access"] + et_times['video'] + et_times['nagivate']
        values.append(x)

    if '#access' in features:
        x = et_times["access"]
        values.append(x)

    if '#video' in features:
        x = et_times["video"]
        values.append(x)

    if '#discussion' in features:
        x = et_times["discussion"]
        values.append(x)

    if '#wiki' in features:
        x = et_times['wiki']
        values.append(x)

    if "#problem" in features:
        x = et_times['problem']
        values.append(x)

    # Number of sessions
    num_session = timeline.split_sessions()
    if '#session' in features:
        x = num_session
        values.append(x)


    # Number of active days
    if '#active_days' in features:
        x = timeline.active_days()
        values.append(x)

    # Number of page views
    if '#page_view' in features:
        x  = 0.0
        for f in ["access","problem","video", "wiki", "discussion"]:
            x += et_times[f]
        values.append(x)

    # average number of video click actions per session
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

    if '#lagging_times' in features:
        x = 0
        for l in lags:
            if l >= 3:
                x += 1.0
        values.append(x)

    if '#avg_lagging' in features:
        if len(lags) > 0:
            avg_lag = sum_lags / len(lags)
        else:
            avg_lag = 0.0
        values.append(avg_lag)

    if '#lagging2week' in features and '#lagging<2week':
        if sum_lags >= 14:
            values.append(1.0)
            values.append(0.0)
        else:
            values.append(0.0)
            values.append(1.0)



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
