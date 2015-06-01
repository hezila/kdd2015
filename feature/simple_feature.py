#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math

from data import *
from util import *

class SimpleFeatureFactory:
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
            cid = dataset.course_by_eid(eid)
            course_strt_dt = dataset.get_course_strt_dt(cid)
            course_end_dt = dataset.get_course_end_dt(cid)
            values = timeline_features(timeline, self.features, self.modules,
                                            course_strt_dt = course_strt_dt,
                                            course_end_dt =course_end_dt)
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

def timeline_features(timeline, features, modules=None, course_strt_dt=None, course_end_dt=None):
    values = []

    f_ids = {}

    d = timeline.duration("d")
    if '#duration' in features:
        values.append(d)

        f_ids['#duration'] = 1

    if '#num_events' in features:
        x = len(timeline.events)
        values.append(x)
        f_ids['#num_events'] = 1

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
        f_ids['#browser_request'] = 1

    if '#browser_access' in features:
        x = browser_et_times["access"]
        values.append(x)
        f_ids['#browser_access'] = 1

    if '#browser_video' in features:
        x = browser_et_times["video"]
        values.append(x)
        f_ids["#browser_video"] = 1

    if "#browser_problem" in features:
        x = browser_et_times['problem']
        values.append(x)
        f_ids['#browser_problem'] = 1

    # if '#browser_close' in features:
    #     x = browser_et_times['page_close']
    #     values.append(x)

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
        f_ids['#server_request'] = 1


    if '#server_access' in features:
        x = server_et_times["access"]
        values.append(x)
        f_ids['#server_access'] = 1

    if '#server_discussion' in features:
        x = server_et_times["discussion"]
        values.append(x)
        f_ids['#server_discussion'] = 1

    if '#server_wiki' in features:
        x = server_et_times['wiki']
        values.append(x)
        f_ids['#server_wiki'] = 1

    if "#server_problem" in features:
        x = server_et_times['problem']
        values.append(x)
        f_ids['#server_problem'] = 1

    if '#server_nagivate' in features:
        x = server_et_times['nagivate']
        values.append(x)
        f_ids['#server_nagivate'] = 1


    if '#num_access' in features:
        x = server_et_times['access'] + browser_et_times['access']
        values.append(x)
        f_ids['#num_access'] = 1

    if '#num_problem' in features:
        x = server_et_times['problem'] + browser_et_times['problem']
        values.append(x)
        f_ids['#num_problem'] = 1

    et_times = timeline.event_times()
    total = 0.01
    for et in et_times.keys():
        if et in EVENT_TYPES:
            total += et_times[et]

    if '#access_pert' in features:
        x = et_times['access'] / total
        values.append(x)
        f_ids['#access_pert'] = 1

    if '#video_pert' in features:
        x = et_times['video'] / total
        values.append(x)
        f_ids['#video_pert'] = 1

    if '#discussion_pert' in features:
        x = et_times['discussion'] / total
        values.append(x)
        f_ids['#discussion_pert'] = 1

    if '#wiki_pert' in features:
        x = et_times['wiki'] / total
        values.append(x)
        f_ids['#wiki_pert'] = 1

    if '#problem_pert' in features:
        x = et_times['problem'] / total
        values.append(x)
        f_ids['#problem_pert'] = 1

    if '#nagivate_pert' in features:
        x = et_times['nagivate'] / total
        values.append(x)
        f_ids['#nagivate_pert'] = 1


    # Number of active days
    if '#active_days' in features:
        x = timeline.active_days()
        values.append(x)
        f_ids['#active_days'] = 1

    if '#active_days_per_week' in features:
        access_dates = set([e.get_timestruct().strftime('%Y%m%d') for e in timeline.events])
        access_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in access_dates]
        weeks = [0 for i in range(82/7+1)]
        start_date = datetime.datetime(2014, 5, 13)
        t = 0.0
        for access_date in access_dates:
            diff = (access_date-start_date).days/7
            if 0 <= diff < len(weeks):
                weeks[diff] += 1
                t += 1.0
        x = t / (len(weeks))
        values.append(x)

        f_ids['#active_days_per_week'] = 1

        for i, week in enumerate(weeks):
            if ('#active_days_week%d' % i) in features:
                x = week
                values.append(x)
                f_ids['#active_days_week%d' % i] = 1

    # Number of sessions
    sessions = timeline.split_sessions()
    num_session = len(sessions)
    if '#session' in features:
        x = num_session
        values.append(x)
        f_ids['#session'] = 1

    if '#avg_requests_per_session' in features:
        x = 0.0
        for s in sessions:
            x += len(s)
        x = x / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_requests_per_session'] = 1


    # average number of video click actions per session
    et_times = timeline.event_times()
    if '#avg_video_per_session' in features:
        x = et_times['video'] / (num_session + 0.0)
        values.append(x)
        f_ids['#avg_video_per_session'] = 1

    if '#avg_discuss_per_session' in features:
        x = et_times['discussion'] / (num_session + 0.0)
        values.append(x)
        f_ids['#avg_discuss_per_session'] = 1

    if '#avg_access_per_session' in features:
        x = et_times['access'] / (num_session + 0.0)
        values.append(x)
        f_ids['#avg_access_per_session'] = 1

    if '#avg_nagivate_per_session' in features:
        x = et_times['nagivate'] / (num_session + 0.0)
        values.append(x)
        f_ids['#avg_nagivate_per_session'] = 1

    if '#avg_problem_per_session' in features:
        x = et_times['problem'] / (num_session + 0.0)
        values.append(x)
        f_ids['#avg_problem_per_session'] = 1

    # Most common request time:
    # night time from 19:00 to 6:59 in the morning and the other half day as day time

    day_times = 0
    night_times = 0

    weekend_times = 0.0
    weekday_times = 0.0

    for e in timeline.events:
        etype = e.get_event()
        if etype not in EVENT_TYPES:
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
        f_ids['#daytime_ratio'] = 1

    if '#night_time' in features and '#day_time' in features:

        if night_times >= day_times:
            values.append(1)
            values.append(0)
        else:
            values.append(0)
            values.append(1)

        f_ids['#night_time'] = 1
        f_ids['#day_time'] = 1

    # most common request week day
    if '#weekend_ratio' in features:

        if (weekend_times + weekday_times) > 0:
            x = weekend_times / (weekend_times + weekday_times + 0.0)
        else:
            x = 0.0
        values.append(x)
        f_ids['#weekend_ratio'] = 1

    if '#weekend_time' in features and '#weekday_time' in features:
        if weekend_times >= weekday_times:
            values.append(1.0)
            values.append(0.0)
        else:
            values.append(0.0)
            values.append(1.0)
        f_ids['#weekend_time'] = 1
        f_ids['#weekday_time'] = 1

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
        f_ids['#active_days_last_week'] = 1

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
        f_ids['#request_last_week'] = 1

    if '#access_last_week' in features:
        x = last_week_event_times['access']
        values.append(x)
        f_ids["#access_last_week"] = 1

    if '#video_last_week' in features:
        x = last_week_event_times['video']
        values.append(x)
        f_ids['#video_last_week'] = 1

    if '#problem_last_week' in features:
        x = last_week_event_times['problem']
        values.append(x)
        f_ids['#problem_last_week'] = 1

    if '#wiki_last_week' in features:
        x = last_week_event_times['wiki']
        values.append(x)
        f_ids['#wiki_last_week'] = 1

    if '#discussion_last_week' in features:
        x = last_week_event_times['discussion']
        values.append(x)
        f_ids['#discussion_last_week'] = 1

    if '#nagivate_last_week' in features:
        x = last_week_event_times['nagivate']
        values.append(x)
        f_ids['#nagivate_last_week'] = 1

    sessions_last_week = timeline.split_sessions(start_day)
    num_session = len(sessions_last_week)

    if '#session_last_week' in features:
        values.append(num_session)
        f_ids['#session_last_week'] = 1

    if '#avg_requests_per_session_last_week' in features:
        x = 0
        for s in sessions_last_week:
            x += len(s)

        x = x / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_requests_per_session_last_week'] = 1

    if '#avg_access_per_session_last_week' in features:
        x = last_week_event_times['access'] / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_access_per_session_last_week'] = 1

    if '#avg_video_per_session_last_week' in features:
        x = last_week_event_times['video'] / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_video_per_session_last_week'] = 1

    if '#avg_problem_per_session_last_week' in features:
        x = last_week_event_times['problem'] / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_problem_per_session_last_week'] = 1

    if '#avg_nagivate_per_session_last_week' in features:
        x = last_week_event_times['nagivate'] / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_nagivate_per_session_last_week'] = 1

    if '#request_last_2week' in features:
        x = last_2week_event_times["access"]
        x += last_2week_event_times['video']
        x += last_2week_event_times['nagivate']
        x += last_2week_event_times['wiki']
        x += last_2week_event_times['problem']
        x += last_2week_event_times['discussion']
        values.append(x)
        f_ids['#request_last_2week'] = 1

    if '#access_last_2week' in features:
        x = last_2week_event_times['access']
        values.append(x)
        f_ids['#access_last_2week'] = 1

    if '#video_last_2week' in features:
        x = last_2week_event_times['video']
        values.append(x)
        f_ids['#video_last_2week'] = 1

    if '#problem_last_2week' in features:
        x = last_2week_event_times['problem']
        values.append(x)
        f_ids['#problem_last_2week'] = 1

    if "#wiki_last_2week" in features:
        x = last_2week_event_times['wiki']
        values.append(x)
        f_ids['#wiki_last_2week'] = 1

    if "#discussion_last_2week" in features:
        x = last_2week_event_times['discussion']
        values.append(x)
        f_ids['#discussion_last_2week'] = 1

    if "#nagivate_last_2week" in features:
        x = last_2week_event_times['nagivate']
        values.append(x)
        f_ids['#nagivate_last_2week'] = 1

    #session_last_2week
    sessions_last_2week = timeline.split_sessions(start_day_2w)
    num_session = len(sessions_last_2week)
    if '#session_last_2week' in features:
        values.append(num_session)
        f_ids['#session_last_2week'] = 1

    if "#avg_requests_per_session_last_2week" in features:
        x = 0.0
        for s in sessions_last_2week:
            x += len(s)
        x = x / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_requests_per_session_last_2week'] = 1

    if "#avg_access_per_session_last_2week" in features:
        x = last_2week_event_times['access'] / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_access_per_session_last_2week'] = 1

    if "#avg_video_per_session_last_2week" in features:
        x = last_2week_event_times['video'] / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_video_per_session_last_2week'] = 1

    if "#avg_problem_per_session_last_2week" in features:
        x = last_2week_event_times['problem'] / (num_session + 0.01)
        values.append(x)
        f_ids['#avg_problem_per_session_last_2week'] = 1

    times_hours = {}
    for h in range(24):
        times_hours[h] = 0.0

    total = 0.0
    for e in timeline.events:
        d, t = e.get_time().split('T')
        h = int(t.split(':')[0])

        etype = e.get_event()
        if etype in EVENT_TYPES:
            times_hours[h] += 1.0
            total += 1.0

    if '#0-6h_request':
        x = sum(map(lambda x: times_hours[x], range(7)))
        if total > 0:
            x = x / total
        values.append(x)
        f_ids['#0-6h_request'] = 1

    if '#6-9h_request':
        x = sum(map(lambda x: times_hours[x], range(6, 10)))
        if total > 0:
            x = x / total
        values.append(x)
        f_ids['#6-9h_request'] = 1

    if '#8-12h_request':
        x = sum(map(lambda x: times_hours[x], range(8, 13)))
        if total > 0:
            x = x / total
        values.append(x)
        f_ids['#8-12h_request'] = 1

    if '#12-18h_request':
        x = sum(map(lambda x: times_hours[x], range(12, 18)))
        if total > 0:
            x = x / total
        values.append(x)
        f_ids['#12-18h_request'] = 1

    if '#17-20h_request':
        x = sum(map(lambda x: times_hours[x], range(17, 20)))
        if total > 0:
            x = x / total
        values.append(x)
        f_ids['#17-20h_request'] = 1

    if '#19-24h_request':
        x = sum(map(lambda x: times_hours[x], range(19, 24)))
        if total > 0:
            x = x / total
        values.append(x)
        f_ids['#19-24h_request'] = 1

    access_hours = set([int(e.get_timestruct().strftime('%H')) for e in timeline.events])
    if '#access_hour_count' in features:
        values.append(len(access_hours))
        f_ids['#access_hour_count'] = 1

    mean_h = 1.0
    if "#access_hour_mean" in features:
        mean_h = reduce(lambda x, y: x+y, access_hours) / (len(access_hours) + 0.0)
        values.append(mean_h)
        f_ids["#access_hour_mean"] = 1

    if '#access_hour_var' in features:
        x = 0.0
        for h in access_hours:
            x += (h - mean_h) * (h - mean_h)
        x = math.sqrt(x / len(access_hours))
        values.append(x)
        f_ids['#access_hour_var'] = 1

    lags = []
    access_dates = sorted(list(set([e.get_time().split('T')[0] for e in timeline.events])))
    access_dates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in access_dates]
    if len(access_dates) == 1:
        lags = [0]
    else:
        lags = [(access_dates[i+1]-access_dates[i]).days for i in range(len(access_dates)-1)]

    start = days[0]
    sum_lags = reduce(lambda x, y: x+y, lags)

    if '#total_lagging' in features:
        values.append(sum_lags)
        f_ids['#total_lagging'] = 1

    if '#lagging_3days_times' in features:
        x = 0
        for l in lags:
            if l >= 3:
                x += 1.0
        values.append(x)
        f_ids['#lagging_3days_times'] = 1

    if '#lagging_1week_times' in features:
        x = 0
        for l in lags:
            if l >= 6:
                x += 1.0
        values.append(x)
        f_ids['#lagging_1week_times'] = 1

    if '#lagging_2week_times' in features:
        x = 0
        for l in lags:
            if l >= 12:
                x += 1.0
        values.append(x)
        f_ids['#lagging_2week_times'] = 1

    if '#max_lagging' in features:
        x = max(lags)
        values.append(x)
        f_ids['#max_lagging'] = 1

    if '#min_lagging' in features:
        x = min(lags)
        values.append(x)
        f_ids['#min_lagging'] = 1

    if '#avg_lagging' in features:
        if len(lags) > 0:
            avg_lag = sum_lags / len(lags)
        else:
            avg_lag = 14
        values.append(avg_lag)
        f_ids['#avg_lagging'] = 1

    if '#std_lagging' in features:
        std = 0.0
        for lag in lags:
            std += (lag - avg_lag) * (lag - avg_lag)
        if len(lags) == 0:
            std = 5
        else:
            std = math.sqrt(std / len(lags))
        values.append(std)
        f_ids['#std_lagging'] = 1

    if '#lagging2week' in features and '#lagging<2week' in features:
        if sum_lags >= 14:
            values.append(1.0)
            values.append(0.0)
        elif sum_lags == 0:
            values.append(1.0)
            values.append(1.0)
        else:
            values.append(0.0)
            values.append(1.0)

        f_ids['#lagging2week'] = 1
        f_ids['#lagging<2week'] = 1

    # access
    x_2d = 0
    x_1w = 0
    x_2w = 0

    days_mod = {}
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
            if d < 0:
                d = 0.0

            if mid not in days_mod:
                days_mod[mid] = [d]
            else:
                days_mod[mid].append(d)

            if d >= 2 * (60 * 60 * 24):
                x_2d += 1.0
            if d >= 7 * (60 * 60 * 24):
                x_1w += 1.0
            if d >= 14 * (60 * 60 * 24):
                x_2w += 1.0


    if '#access_2day_lagging' in features:
        values.append(x_2d)
        f_ids['#access_2day_lagging'] = 1

    if "#access_1week_lagging" in features:
        values.append(x_1w)
        f_ids['#access_1week_lagging'] = 1

    if '#access_2week_lagging' in features:
        values.append(x_2w)
        f_ids['#access_2week_lagging'] = 1

    days_acs1_mod_rls = []
    days_acslst_mod_rls = []
    for mid in days_mod.keys():
        ds = days_mod[mid]
        ds.sort()
        days_acs1_mod_rls.append(ds[0] / (60 * 60 * 24.0))
        days_acslst_mod_rls.append(ds[-1] / (60 * 60 * 24.0))

    if '#median_days_acs1_mod_rls' in features:
        x = 7
        if len(days_acs1_mod_rls) > 0:
            x = median(days_acs1_mod_rls)
        values.append(x)
        f_ids['#median_days_acs1_mod_rls'] = 1

    if "#25pert_days_acs1_mod_rls" in features:
        x = 7
        if len(days_acs1_mod_rls) > 0:
            x = median(days_acs1_mod_rls)
        values.append(x)
        f_ids['#25pert_days_acs1_mod_rls'] = 1

    if '#75pert_days_acs1_mod_rls' in features:
        x = 7
        if len(days_acs1_mod_rls) > 0:
            x = median(days_acs1_mod_rls)
        values.append(x)
        f_ids['#75pert_days_acs1_mod_rls'] = 1

    if '#median_days_acslst_mod_rls' in features:
        x = 7
        if len(days_acslst_mod_rls) > 0:
            x = median(days_acslst_mod_rls)
        values.append(x)
        f_ids['#median_days_acslst_mod_rls'] = 1

    if "#25pert_days_acslst_mod_rls" in features:
        x = 7
        if len(days_acslst_mod_rls) > 0:
            x = median(days_acslst_mod_rls)
        values.append(x)
        f_ids['#25pert_days_acslst_mod_rls'] =1

    if '#75pert_days_acslst_mod_rls' in features:
        x = 7
        if len(days_acslst_mod_rls) > 0:
            x = median(days_acslst_mod_rls)
        values.append(x)
        f_ids['#75pert_days_acslst_mod_rls'] = 1

    strt = None
    end = None
    for e in timeline.events:
        t = e.get_stamp()
        if strt == None:
            strt = t
            end = t
        if strt > t:
            strt = t
        if end < t:
            end = t


    if '#days_course_strt_access' in features:
        x = strt - course_strt_dt
        x = x / (60 * 60 * 24.0)
        values.append(x)
        f_ids['#days_course_strt_access'] = 1

    if '#days_course_end_access_lst' in features:
        x = course_end_dt - end
        x = x / (60 * 60 * 24.0)
        values.append(x)
        f_ids['#days_course_end_access_lst'] = 1

    access_dates = [(e.get_timestruct().strftime('%Y%m%d'), e.get_timestruct()) for e in timeline.events]
    daily_logs = {}
    for d in access_dates:
        if d[0] in daily_logs.keys():
            daily_logs.get(d[0]).append(d[1])
        else:
            daily_logs[d[0]] = [d[1]]
    staytimes = [(max(v)-min(v)).seconds for k, v in daily_logs.items()]

    if '#staytime_min' in features:
        x = min(staytimes)
        values.append(x)
        f_ids['#staytime_min'] = 1

    if '#staytime_max' in features:
        x = max(staytimes)
        values.append(x)
        f_ids['#staytime_max'] = 1

    stay_mean = 0.0
    if '#staytime_mean' in features:
        stay_mean = sum(staytimes) / (len(staytimes) + 0.0)
        values.append(stay_mean)
        f_ids['#staytime_mean'] = 1

    if '#staytime_var' in features:
        x = 0.0
        for v in staytimes:
            x = (v - stay_mean) * (v - stay_mean)
        x = math.sqrt(x / (len(staytimes) + 0.0))
        values.append(x)
        f_ids['#staytime_var'] = 1

    x = 0
    s = 0
    for e in timeline.events:
        es = e.get_source()
        if 'browser' == es:
            x += 1.0
        s += 1.0

    # browser vs server
    if '#ratio_browser' in features:
        values.append(x / s)
        f_ids['#ratio_browser'] = 1

    if '#server_cnt' in features:
        values.append(s - x)
        f_ids['#server_cnt'] = 1

    start_date = datetime.datetime(2014, 5, 13)
    event_week = {}
    for event_type in EVENT_TYPES:
        event_week[event_type] = [0 for i in range(82/7+1)]
    targets = set(['{0},{1}'.format(e.get_timestruct().strftime('%Y%m%d'), e.get_event()) for e in timeline.events])
    for target in targets:
        d, e = target.split(',')
        if e not in EVENT_TYPES:
            continue
        diff = (datetime.datetime.strptime(d, '%Y%m%d')-start_date).days/7
        if 0 <= diff < len(event_week[e]):
            event_week[e][diff] += 1
    for event in EVENT_TYPES:
        weeks = event_week[event]
        for i, week in enumerate(weeks):
            k = '#event_days_%s_week%d' % (event, i)
            if k in features:
                values.append(week)
                f_ids[k] = 1

    start_date = datetime.datetime(2014, 5, 13)
    weeks = [0 for i in range(82/7+1)]

    for i in range(len(timeline.events)-1):
        if timeline.events[i].get_event() != 'video':
            continue
        time_delta = timeline.events[i+1].get_timestruct()-timeline.events[i].get_timestruct()

        if 600 < time_delta.seconds < 18000 and time_delta.days == 0:
            diff = (timeline.events[i+1].get_timestruct()-start_date).days/7
            if 0 <= diff < len(weeks):
                weeks[diff] += 1

    for i, week in enumerate(weeks):
        k = '#video_over10minutes_week%d' % i
        if k in features:
            values.append(week)
            f_ids[k] = 1

    weeks = [0 for i in range(82/7 + 1)]

    for i in range(len(timeline.events)-1):
        if timeline.events[i].get_event() != 'problem':
            continue
        time_delta = timeline.events[i+1].get_timestruct()-timeline.events[i].get_timestruct()

        if 180 < time_delta.seconds < 18000 and time_delta.days == 0:
                diff = (timeline.events[i+1].get_timestruct()-start_date).days/7
                if 0 <= diff < len(weeks):
                    weeks[diff] += 1

    for i, week in enumerate(weeks):
        k = '#problem_over3minutes_week%d' % i
        if k in features:
            values.append(week)
            f_ids[k] = 1
    # print "LEN: %d" % len(values)
    # for fk in features:
    #     if fk not in f_ids:
    #         print fk
    return values
