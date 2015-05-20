#/usr/bin/env python
#-*- coding: utf-8 -*-

from optparse import OptionParser

from data import *
from util import *

import plotly.plotly as py
from plotly.graph_objs import *

def dropout_bar(dataset):

    pos_bar = {}
    pos_bar['x'] = []
    pos_bar['y'] = {}
    neg_bar = {}
    neg_bar['x'] = []
    neg_bar['y'] = {}

    for cid in dataset.iter_courses():
        pos_bar['x'].append(cid)
        neg_bar['x'].append(cid)
        pos_bar['y'][cid] = 0
        neg_bar['y'][cid] = 0
        for eid in dataset.iter_eids_by_course(cid):
            label = dataset.get_label(eid)

            if label == '1':
                pos_bar['y'][cid] += 1.0
            else:
                neg_bar['y'][cid] += 1.0

    trace1 = Bar(
        x=pos_bar['x'],
        y=[pos_bar['y'][k] for k in pos_bar['x']],
        name='Dropout'
    )
    trace2 = Bar(
        x=neg_bar['x'],
        y=[neg_bar['y'][k] for k in neg_bar['x']],
        name='Keep'
    )
    data = Data([trace1, trace2])
    layout = Layout(
        barmode='stack'
    )
    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='dropout-bar')


def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)
    parser.add_option("-e", "--enrollment", dest="erlm",
        help="the enrollments file")
    parser.add_option("-l", "--log", dest="log",
        help="the log file")
    parser.add_option("-t", "--truth", dest="truth",
        help="the truth file")

    parser.add_option("-s", "--test_enrollment", dest="test_erlm",
        help="the test enrollment file")
    parser.add_option("-k", "--test_log", dest="test_log",
        help="the test log file")

    (options, remainder) = parser.parse_args()

    print 'Traing data ....'
    train = load_data(options.erlm, options.log, options.truth)
    print '#################'
    print
    print "Test data ..."
    test = load_data(options.test_erlm, options.test_log)
    print '#################'


    new_users = []
    for uid in test.iter_users():
        if not train.has_user(uid):
            new_users.append(uid)
    print '!!! %d new users in test dataset' % (len(new_users))

    new_courses = []
    for cid in test.iter_courses():
        if not train.has_course(cid):
            new_courses.append(cid)
    print '!!! %d new courses in test dataset' % len(new_courses)

    # print 'Hist: users vs enrollment'
    # user_hist = []
    # for uid in train.iter_users():
    #     eids = train.eids_by_user(uid)
    #     user_hist.append(len(eids))
    #
    # output = open("user_hist.csv", 'w')
    # for v in user_hist:
    #     output.write('%d\n' % v)
    # output.close()
    #
    # print 'Hist: course vs enrollment'
    # course_hist = []
    # for cid in train.iter_courses():
    #     eids = train.eids_by_course(cid)
    #     course_hist.append(len(eids))
    # output = open("course_hist.csv", 'w')
    # for v in course_hist:
    #     output.write("%d\n" % v)
    # output.close()


    dropout_bar(train)

if __name__ == '__main__':
    main()
