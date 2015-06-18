#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math
import random
import time
import datetime
from itertools import *


import logging

import functools


# given an iterable of pairs return the key corresponding to the greatest value
def argmax(pairs):
    return max(pairs, key=lambda x: x[1])[0]

# given an iterable of values return the index of the greatest value
def argmax_index(values):
    return argmax(enumerate(values))

# given an iterable of keys and a function f, return the key with largest f(key)
def argmax_f(keys, f):
    return max(keys, key=f)

def countties(Xranks, issorted = True, ztol = 1.0e-5):
	"""
	Returns an array of pairs (n, x) where n is the tied count and x is the
	tied value.
	"""
	tiescount = []
	if not issorted:
		X = sorted(Xranks)
	else:
		X = Xranks
	x	  = X[0]
	ncount = 1
	n	  = len(X)
	for j in range(1, n):
	  if abs(X[j] - x)< ztol:
		 ncount += 1
	  else:
		if ncount > 1:
		   tiescount.append((ncount, x))
		ncount = 1
		x = X[j]
	# last pair value
	if ncount > 1:
	   tiescount.append((ncount, x))

	return tiescount

def kendall(X,Y, ztol = 1.0e-5):
	"""
	Computes the Kendall tau correlation coefficient for input
	ordinal data X and Y.
	"""
	n = len(X)

	xi = [(x,i) for i, x in enumerate(X)]
	xi.sort()
	yi = [Y[i] for (x,i) in xi]
	L, G = 0, 0

	# count ties.
	tx = countties(X, issorted=True)
	ty = countties(yi)
	Tx = sum([i*(i-1) for (i, x) in tx])*0.5
	Ty = sum([i*(i-1) for (i, x) in ty])*0.5
	for i in range(n):
		# Count number of < and > ranked data for the corresponding y elements.
		l, g = 0, 0
		ycmp = yi[i]
		for j in range(i+1, n):
			if  yi[j] > ycmp:
			   g += 1
			elif yi[j] < ycmp:
			   l += 1
		L+= l
		G+= g
	f =  0.5 * n * (n-1)
	den = math.sqrt((f - Tx)* (f - Ty))

	tau = (G - L)/ den
	return L, G, tau


def percentile(N, percent, key=lambda x:x):
    """
    Find the percentile of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile of the values
    """
    if not N:
        return None
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0+d1

# median is 50th percentile.
median = functools.partial(percentile, percent=0.5)

# Formated current timestamp
def current_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# Log message with timestamp
def log_info(message):
    ts = time.time()
    logging.info(message + " " + current_timestamp())

# Initialise logging
def init_logging(log_file_path):
    logging.basicConfig(format='%(message)s', level=logging.INFO, filename=log_file_path)

def make_submission(m, test, filename="submission.csv"):
    pass

def get_duration(start_time_string, end_time_string, format="%Y-%m-%dT%H:%M:%S"):

    start_time_stamp = get_timestamp(start_time_string, format=format)
    end_time_stamp = get_timestamp(end_time_string, format = format)

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
