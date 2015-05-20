#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math
import random
import time
import datetime
from itertools import *

from data import *


def load_data(erlm_file, event_file, label_file=None):
    print 'loading enrollments ...'
    erlms = Enrollments()
    i = 0
    with open(erlm_file, 'r') as r:
        for line in r:
            if i == 0:
                i += 1
                continue
            eid, uid, cid = line.strip().split(',')

            erlms.add(uid, cid, eid)

            # i += 1
            # if i % 1000000000:
            #     print '#',
    print
    print 'end of loadings enrollments!'
    print 'Total users: %d' % len(erlms.users)
    print 'Total course: %d' % len(erlms.courses)
    print 'Total size: %d' % erlms.size
    print

    print 'loading event dataset ...'
    event_dataset = EventDateset(erlms)
    i = 0
    with open(event_file, 'r') as r:
        # headers: enrollment_id,time,source,event,object
        for line in r:
            if i == 0:
                i += 1
                continue

            eid, time, source, event, obj = line.strip().split(',')
            event = Event(eid, time, source, event, obj)

            event_dataset.add_event(event)
            # i += 1
            # if i % 1000000000 == 0:
            #     print '#',
    print '> sort event time line'
    event_dataset.sort_timeline()
    print '> end of sort'
    print 'end of loadings dataset!'
    print 'Total events: %d' % event_dataset.size
    print

    if label_file is None:
        return event_dataset

    print 'loading truth labels ...'
    labels = {} #  for a dropout event and 0 for continuing study
    with open(label_file, 'r') as r:
        for line in r:
            eid, label = line.strip().split(',')
            labels[eid] = label
    print 'end of loading labels!'
    print 'Total train size: %d' % len(labels)
    event_dataset.set_labels(labels)

    return event_dataset

def make_submission(m, test, filename="submission.csv"):
    pass

def get_duration(start_time_string, end_time_string):

    start_time_stamp = get_timestamp(start_time_string)
    end_time_stamp = get_timestamp(end_time_string)

    duration = end_time_stamp - start_time_stamp

    # what's the next step to get duration time, like x minute?
    # duration_minutes = duration.total_seconds() / 60
    return duration


def get_timestamp(s, format="%Y-%m-%dT%H:%M:%S"):
    """
    Get the time stampt from the time string, e.g., 2013-12-22T14:55:28
    """
    return time.mktime(datetime.datetime.strptime(s, format).timetuple())

def get_date(timestamp):
    """
    Get the date from the timestamp
    :param timestamp: the timestamp
    :returns: the string represention of the date
    """
    value = datetime.datetime.fromtimestamp(timestamp)
    return value.strftime('%Y-%m-%d %H:%M:%S')

def order(x):
    """
    returns the order of each element in x as a list.
    """
    L = len(x)
    rangeL = range(L)
    z = izip(x, rangeL)
    z = izip(z, rangeL)  # avoid problems with duplicates.
    D = sorted(z)
    return [d[1] for d in D]


def order_dict(d):
    """
    returns the order of each item in d as a dict
    """
    k_list = []
    v_list = []
    for k, v in d.items():
        k_list.append(k)
        v_list.append(v)

    orders = order(v_list)
    return [k_list[o] for o in orders]


def rank(x):
    """
    Returns the rankings of elements in x as a list.
    """
    L = len(x)
    ordering = order(x)
    ranks = [0] * len(x)
    for i in range(L):
        ranks[ordering[i]] = i
    return ranks
