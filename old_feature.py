#!/usr/bin/env python
#-*- coding: utf-8 -*-

from data import *
from util import *

class TimelineFeatureFactory:
    def __init__(self):
        fnames = ['duration', 'active_days'] + map(lambda x: '%s_times' % x, EVENT_TYPES)
        names = []
        for n in fnames:
            names.append('%s_invert_log' % n)
            names.append('%s_log' % n)
        self.fnames = names + ['avg_lag', "avg_lag_log", "std_lag", "std_lag_log"]

    def gen_features(self, dataset):
        for timeline in dataset.iter_timelines():
            eid = timeline.eid

            x = timeline_features(timeline)
            values = map(lambda k: x[k], self.fnames)
            target = dataset.get_label(eid)
            yield (eid, target, values)

    def dump(self, dataset, filename):
        output = open(filename, 'w')
        if dataset.isTest():
            output.write("eid,%s\n" % ','.join(self.fnames))
        else:
            output.write("eid,target,%s\n" % ','.join(self.fnames))

        for eid, target, values in self.gen_features(dataset):
            if not dataset.isTest():
                output.write('%s,%s,%s\n' % (eid, target, (','.join(['%.3f' % v for v in values]))))
            else:
                output.write('%s,%s\n' % (eid, (','.join(['%.3f' % v for v in values]))))
        output.close()

def timeline_features(timeline):
    # ['problem', 'wiki', 'access', 'nagivate', 'discussion', 'video']
    # 2 - 7
    x = {}
    d = timeline.duration("d") - 7 # ignore the first week
    if d < 0:
        d = 0
    x["duration_invert_log"] = invert_log(d)
    x["duration_log"] = count_log(d)

    d = timeline.active_days() - 1 # ignore the first day

    x['active_days_invert_log'] = invert_log(d)
    x['active_days_log'] = count_log(d)

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

    if len(lags) > 0:
        avg_lag = sum_lags / len(lags)
    else:
        avg_lag = 7

    std = 0.0
    for l in lags:
        std += (l - avg_lag) * (l - avg_lag)

    if len(lags) > 0:
        std = math.sqrt(std / len(lags))
    else:
        std = 0

    x['avg_lag'] = avg_lag
    x['avg_lag_log'] = count_log(avg_lag)
    x['std_lag'] = std
    x['std_lag_log'] = count_log(std)

    et_times = timeline.event_times()
    for et in EVENT_TYPES:
        # if et in ['problem', 'access', 'nagivate']:
        #     x["%s_times" % et] = invert_log(et_times[et])
        # else:
        #     x["%s_times" % et] = count_log(et_times[et])

        x["%s_times_invert_log" % et] = invert_log(et_times[et])
        x["%s_times_log" % et] = count_log(et_times[et])

    return x
