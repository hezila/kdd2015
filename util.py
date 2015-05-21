#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math
import random
import time
import datetime
from itertools import *

def make_submission(m, test, filename="submission.csv"):
    pass

def get_duration(start_time_string, end_time_string):

    start_time_stamp = get_timestamp(start_time_string)
    end_time_stamp = get_timestamp(end_time_string)

    duration = end_time_stamp - start_time_stamp

    # what's the next step to get duration time, like x minute?
    # duration_minutes = duration.total_seconds() / 60
    return duration

def get_weekday(time_string, format="%Y-%m-%dT%H:%M:%S"):
    return time.strftime("%A", time.strptime(time_string, format))

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


def duration_str(d):
    sb = ""
    week = int(d / (60 * 60 * 24 * 7))
    if week > 0:
        d -= week * (60 * 60 * 24 * 7)
        sb += "%d weeks " % week

    day = int(d / (60 * 60 * 24))
    if day > 0:
        d -= day * (60 * 60 * 24)
        sb += "%d days " % day

    hour = int(d / (60 * 60))
    if hour > 0:
        d -= hour * (60 * 60)
        sb += "%d hours " % hour

    minute = int (d / 60)
    if minute > 0:
        d -= minute * 60
        sb += "%d minutes " % minute

    sb += "%f seconds" % d

    return sb

def count_log(x):
    return math.log(1 + x)

def invert_log(x):
    return 1.0 / (1 + math.log(1 + x))

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
