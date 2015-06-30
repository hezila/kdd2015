#!/usr/bin/env python
#-*- coding: utf-8 -*-

import itertools
import os
import datetime

from utils import *


__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-09-2015'

class Module:
    def __init__(self, cid, mid, cate, children, start):
        self.cid = cid
        self.mid = mid
        self.cate = cate
        self.start = start
        self.children = children

    def get_cid(self):
        return self.cid

    def get_mid(self):
        return self.mid

    def get_cate(self):
        return self.cate

    def get_start(self):
        return self.start

    def get_children(self):
        return self.children


class ModuleDB:
    def __init__(self):
        self.cate_by_mid = {}
        self.modules = {}
        self.modules_by_cid = {}
        self.parent_by_mid = {}

    def add(self, module):
        mid = module.get_mid()
        cate = module.get_cate()
        cid = module.get_cid()

        self.modules[mid] = module

        self.cate_by_mid[mid] = cate
        self.modules_by_cid.setdefault(cid, []).append(mid)
        for c in module.get_children():
            self.parent_by_mid[c] = mid
            if c not in self.modules_by_cid[cid]:
                self.modules_by_cid[cid].append(c)

    def exist(self, mid):
        return (mid in self.modules)

    def order_modules(self):
        for cid in self.modules_by_cid.keys():
            modules = [(mid, self.get_start(mid)) for mid in self.modules_by_cid[cid] if self.get_start(mid) is not None]
            modules = sorted([(m, t) for m, t in modules if self.get_cate(m) in ['chapter', 'sequential']], key=lambda x: x[1])
            self.modules_by_cid[cid] = [m for m, t in modules]

    def get_rank(self, cid, mid):
        ms = self.modules_by_cid[cid]
        if mid in ms:
            return ms.index(mid) + 1
        else:
            return None

    def get_cate(self, mid):
        m = self.modules[mid]
        return m.get_cate()

    def get_parent(self, mid):
        if mid in self.parent_by_mid:
            return self.parent_by_mid[mid]
        return None

    def get_start(self, mid):
        if mid not in self.modules:
            return None
        m = self.modules[mid]

        start = m.get_start()
        if start is None:
            p = self.get_parent(m)
            if p and self.get_cate(p) in ['chapter', 'sequential']:
                start = self.get_start(p)
                if start is None:
                    p = self.get_parent(p)
                    if p and self.get_cate(p) in ['chapter', 'sequential']:
                        start = self.get_start(p)
                    return start
                else:
                    return start
            else:
                return start
        else:
            return start

def load_modules(filename):
    modules = ModuleDB()
    with open(filename, 'r') as r:
        for line in r:
            line = line.strip()
            cid, mid, cate, children, start = line.split(',')
            # course_id,module_id,category,children,start
            if cate not in ['chapter', 'sequential']:
                continue

            children = children.strip()
            if len(children) > 0:
                children = children.split(' ')
            else:
                children = []
            if len(start) == 0 or start in ['null', 'start']:
                start = None
            else:
                start = datetime.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S')
            m = Module(cid, mid, cate, children, start)
            modules.add(m)
    return modules
