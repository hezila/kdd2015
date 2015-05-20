#!/usr/bin/env python
#-*- coding: utf-8 -*-

import time
import datetime

from util import *

class Enrollments:
    def __init__(self):
        self.users = []
        self.courses = []
        self.eids_by_user = {}
        self.eids_by_course = {}
        self.erlms = {}
        self.size = 0

    def user_by_eid(self, eid):
        uid, cid = self.erlms[eid]
        return uid

    def course_by_eid(self):
        uid, cid = self.erlms[eid]
        return cid

    def add(self, uid, cid, eid):
        if uid not in self.users:
            self.users.append(uid)

        if cid not in self.courses:
            self.courses.append(cid)

        self.erlms[eid] = (uid, cid)
        if uid not in self.eids_by_user:
            self.eids_by_user[uid] = [eid]
        else:
            self.eids_by_user[uid].append(eid)

        if cid not in self.eids_by_course:
            self.eids_by_course[cid] = [eid]
        else:
            self.eids_by_course[cid].append(eid)

        self.size += 1

    def get_eids_by_user(self, uid, dft=[]):
        if uid in self.eids_by_user:
            return self.eids_by_user[uid]
        return dft

    def get_eids_by_course(self, cid, dft=[]):
        if cid in self.eids_by_course:
            return self.eids_by_course[cid]
        return dft


class Event:
    """
    Each event contains the following information
        - enrollment_id - Enrollment ID.
        - time - Time of the event.
        - source - Event source (server or browser).
        - event - In terms of event type, we defined 7 different event types:
              1. problem - Working on course assignments.
              2. video - Watching course videos.
              3. access - Accessing other course objects except videos and assignments.
              4. wiki - Accessing the course wiki.
              5. discussion - Accessing the course forum.
              6. navigate - Navigating to other part of the course.
              7. page_close – Closing the web page.
        - object - The object the student access or navigate to.(For navigate and access event only).
    """
    def __init__(self, eid, st, source, event, obj=None):
        self.eid = eid #enrollment id
        self.time = st
        self.stamp = time.mktime(datetime.datetime.strptime(st, "%Y-%m-%dT%H:%M:%S").timetuple())
        self.source = source
        self.event = event
        self.obj = obj

    def get_eid(self):
        return self.eid

    def get_time(self):
        return self.time

    def get_stamp(self):
        return self.stamp

    def get_source(self):
        return self.source

    def get_event(self):
        return self.event

    def get_obj(self):
        return self.obj

class EventTimeLine:
    def __init__(self, eid=None, events = []):
        self.eid = eid
        self.size = len(events)
        self.events = events
        self.ready = False
        self.events_by_day = {}
        for i, e in enumerate(events):
            ymd = e.get_time().split('T')[0]
            if ymd not in self.events_by_day:
                self.events_by_day[ymd] = [i]
            else:
                self.events_by_day[ymd].append(i)

    def add_event(self, event):
        self.events.append(event)

        ymd = event.get_time().split('T')[0]
        if ymd not in self.events_by_day:
            self.events_by_day[ymd] = [self.size]
        else:
            self.events_by_day[ymd].append(self.size)
        self.size += 1


    def sort(self):
        self.events.sort(key=lambda e: e.get_stamp())
        self.ready = True

    def duration(self, unit="s"):
        if not self.ready:
            self.sort()

        if self.size == 0:
            return 0

        duration = get_duration(self.events[0].get_time(), self.events[-1].get_time())
        if unit == 's':
            return duration
        elif unit == 'm':
            return duration / 60
        elif unit == 'h':
            return duration / (60 * 60)
        elif unit == 'd':
            return duration / (60 * 60 * 24)
        elif unit == 'w':
            return duration / (60 * 60 * 24 * 7)

        return duration

    def pretty_duration(self):
        d = self.duration()
        return duration_str(d)


    def display(self):
        if not self.ready:
            self.sort()
        last_event = None
        for i, e in enumerate(self.events):
            print '%d: %s at (%s) %s' % (i, e.get_event(), get_weekday(e.get_time()), e.get_time())
            if last_event is not None:
                d = get_duration(last_event.get_time(), e.get_time())
                print ">>>lag: %s" % duration_str(d)
        print "Duration: (%d days) %s" % (len(self.events_by_day), self.pretty_duration())

class EventDateset:
    """
    Dataset for course related events
    """
    def __init__(self, erlm):
        self.timeline_by_eid = {}
        self.erlm = erlm # enrollment matrix
        self.size = 0
        self.labels = None

    def sort_timeline(self):
        for eid in self.timeline_by_eid:
            self.timeline_by_eid[eid].sort()
            self.timeline_by_eid[eid].display()

    def set_labels(self, labels):
        self.labels = labels

    def get_label(self, eid):
        return self.labels[eid]

    def add_event(self, event):
        eid = event.get_eid()
        if eid not in self.timeline_by_eid:
            self.timeline_by_eid[eid] = EventTimeLine(eid, [event])
        else:
            self.timeline_by_eid[eid].add_event(event)
        self.size += 1

    def get_events_by_eid(self, eid):
        return self.timeline_by_eid[eid].events

    def iter_users(self):
        for uid in self.erlm.users:
            yield uid

    def has_user(self, uid):
        return (uid in self.erlm.users)

    def iter_courses(self):
        for cid in self.erlm.courses:
            yield cid

    def has_course(self, cid):
        return (cid in self.erlm.courses)

    def iter_event(self):
        for e in self.events:
            yield e

    def course_by_eid(self, eid):
        return self.erlm.course_by_eid(eid)

    def user_by_eid(self, eid):
        return self.erlm.user_by_eid(eid)

    def eids_by_user(self, uid):
        eids = self.erlm.get_eids_by_user(uid)
        return eids

    def eids_by_course(self, cid):
        eids = self.erlm.get_eids_by_course(cid)
        return eids

    def iter_eids_by_course(self, cid):
        for eid in self.erlm.get_eids_by_course(cid):
            yield eid

    def courses_by_user(self, uid):
        eids = self.eids_by_user(uid)
        cids = []
        for eid in eids:
            cid = self.course_by_eid(eid)
            if cid not in cids:
                cids.append(cid)

        return cids

    def users_by_course(self, cid):
        eids = self.eids_by_course(cid)
        uids = []
        for eid in eids:
            uid = self.user_by_eid(eid)
            if uid not in uids:
                uids.append(uid)
        return uids
