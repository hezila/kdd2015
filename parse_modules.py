#/usr/bin/env python
#-*- coding: utf-8 -*-

from optparse import OptionParser

from data import *
from util import *


def load_modules(filename):
    modules = Modules()
    with open(filename, 'r') as r:
        for line in r:
            line = line.strip()
            cid, mid, cate, children, start = line.split(',')
            # course_id,module_id,category,children,start
            children = children.strip()
            if len(children) > 0:
                children = children.split(' ')
            else:
                children = []
            m = Module(cid, mid, cate, children, start)
            modules.add_module(m)
    return modules

def load_enrollments(filename):
    erlms = Enrollments()
    i = 0
    with open(filename, 'r') as r:
        for line in r:
            if i == 0:
                i += 1
                continue
            eid, uid, cid = line.strip().split(',')

            erlms.add(uid, cid, eid)

            # i += 1
            # if i % 1000000000:
            #     print '#',
    # print
    # print 'end of loadings enrollments!'
    # print 'Total users: %d' % len(erlms.users)
    # print 'Total course: %d' % len(erlms.courses)
    # print 'Total size: %d' % erlms.size
    return erlms

def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)
    parser.add_option("-m", "--modules", dest="modules",
        help="the modules file")
    parser.add_option("-e", "--enrollment", dest="erlm",
        help="the enrollments file")
    parser.add_option("-l", "--log", dest="log",
        help="the log file")

    (options, remainder) = parser.parse_args()

    modules = load_modules(options.modules)

    erlms = load_enrollments(options.erlm)

    i = 0
    with open(options.log, 'r') as r:
        for line in r:
            if i == 0:
                i += 1
                continue
            line = line.strip()
            # enrollment_id,time,source,event,object
            eid, tm, source, event, mid = line.split(',')
            print [source, event, mid]
            if modules.exist(mid):
                m = modules.get_module_by_mid(mid)
                print '%s\t%s\t' % (event, m.get_cate())
                # cdn = m.get_children()
                # print [modules.get_module_by_mid(c).get_cate() for c in cdn]

            else:
                if mid in erlms.courses:
                    print '%s\t>>course: (%s)<<' % (event, mid)
                else:
                    print '%s\t>>!!! %s !!!<<' % (event, mid)

if __name__ == '__main__':
    main()
