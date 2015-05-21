#!/usr/bin/env python
#-*- coding: utf-8 -*-

from data import *
from util import *

class TimelineFeatureFactory:
    def __init__(self):
        self.fnames = ['duration', 'active_days'] + map(lambda x: '%s_times' % x, EVENT_TYPES)

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
    d = timeline.duration("d")
    # x["duration"] = invert_log(d)
    x["duration"] = count_log(d)

    d = timeline.active_days()
    x['active_days'] = invert_log(d)
    # x['active_days'] = count_log(d)

    et_times = timeline.event_times()
    for et in EVENT_TYPES:
        if et in ['problem', 'access', 'nagivate']:
            x["%s_times" % et] = invert_log(et_times[et])
        else:
            x["%s_times" % et] = count_log(et_times[et])

    return x
