#/usr/bin/env python
#-*- coding: utf-8 -*-

from optparse import OptionParser

from data import *
from util import *
from feature.simple_feature import SimpleFeatureFactory

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



def load_data(erlm_file, event_file, label_file=None):
    print 'loading enrollments ...'
    erlms = load_enrollments(erlm_file)
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
    # print '> sort event time line'
    # event_dataset.sort_timeline()
    # print '> end of sort'
    print 'end of loadings dataset!'

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


def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)
    parser.add_option("-e", "--enrollment", dest="erlm",
        help="the enrollments file")
    parser.add_option("-m", "--modules", dest="modules",
        help="the modules file")
    parser.add_option("-l", "--log", dest="log",
        help="the log file")
    parser.add_option("-f", "--feature", dest="feature",
        help="the feature file")
    parser.add_option("-t", "--truth", dest="truth",
        help="the truth file")

    parser.add_option("-s", "--test_enrollment", dest="test_erlm",
        help="the test enrollment file")
    parser.add_option("-k", "--test_log", dest="test_log",
        help="the test log file")

    (options, remainder) = parser.parse_args()

    print 'loading modules'
    modules = load_modules(options.modules)
    print 'end of loading modules!'

    print 'Traing data ....'
    train = load_data(options.erlm, options.log, options.truth)
    print '#################'
    print
    print "Test data ..."
    test = load_data(options.test_erlm, options.test_log)
    print '#################'


    ftrs = []
    with open(options.feature, 'r') as r:
        for line in r:
            line = line.strip()
            if len(line) == 0 or line.startswith('%'):
                continue

            if ':' not in line:
                fname = line
            else:
                fname, _ = line.split(':')
                fname = fname.strip()

            if fname in ftrs:
                print 'Oops: please check the features you attemp to extract!!!'
                sys.exit(0)

            ftrs.append(fname)

    feature_factory = SimpleFeatureFactory(ftrs, modules)

    feature_factory.dump(train, 'simple_train.csv')
    feature_factory.dump(test, 'simple_test.csv')

if __name__ == '__main__':
    main()
