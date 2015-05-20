#!/usr/bin/env python
#-*- coding: utf-8 -*-

from data import *

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
            if not dataset.isTest:
                output.write('%s,%s,%s\n' % (eid, target, ','.join('%.d' % v for v in values)))
            else:
                output.write('%s,%s\n' % (eid, ','.join('%.d' % v for v in values)))
        output.close()

def timeline_features(timeline):
    x = {}
    x["duration"] = timeline.duration("d")

    x['active_days'] = timeline.active_days()

    et_times = timeline.event_times()
    for et in EVENT_TYPES:
        x["%s_times" % et] = et_times[et]

    return x
