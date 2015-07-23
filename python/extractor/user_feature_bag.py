#!/usr/bin/env python
#-*- coding: utf-8 -*-

from feature_bag import FeatureBag
import datetime
import time
import os
import math

#import numpy as np

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__='02-07-2015'

event_types = ['problem', 'video', 'access', 'wiki', 'discussion', 'nagivate', 'page_close']


class UserFeatureBag(FeatureBag):

    def __init__(self, enrollment_id, logs, feature_keys, feature_values):
        FeatureBag.__init__(self, enrollment_id, logs, feature_keys, feature_values)
        self.logs = sorted(logs, key = lambda x: x['time'])

    def extract_user_features(self, db):
        uid = self.logs[0]['user_name']
        cnt = 0.0
        if uid in db.courses_by_user:
            cnt = len(db.courses_by_user[uid]) + 0.0
        #self.feature_keys.append('user_courses_count')
        #self.feature_values.append(cnt)

        user_drops = 0.0
        user_drop_ratio = 0.8
        if uid in db.user_drops:
            user_drops = db.user_drops[uid] - 1.0
            user_drop_ratio = db.user_drops_ratio[uid]
        #self.feature_keys.append('user_drops_count')
        #self.feature_values.append(user_drops)
        self.feature_keys.append('user_drop_ratio_scale')
        self.feature_values.append(user_drop_ratio)
        return self
