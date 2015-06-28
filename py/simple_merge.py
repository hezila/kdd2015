#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys, os
import math

path1 = "newfeature_submission.csv"
path2 = "azuer_result.csv"

paths = [path1, path2]

results = {}
ids = []

for path in paths:
    with open(path, 'r') as r:
        for line in r:
            eid, prob = line.strip().split(',')
            eid = int(eid)
            results.setdefault(eid, []).append(float(prob))

for eid in results.keys():
    p1, p2 = results[eid]
    results[eid] = math.sqrt(p1 * p2)
    ids.append(eid)

ids = sorted(ids)

with open('merge_submission.csv', 'w') as output:
    for eid in ids:
        p = results[eid]
        output.write("%s,%f\n" % (eid, p))
    output.close()

            
