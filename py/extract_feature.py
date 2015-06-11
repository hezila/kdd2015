#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys

from feature.enrollment_feature_extractor import EnrollmentFeatureExtractor
from feature.course_feature_extractor import CourseFeatureExtractor
# from feature.module_feature_extractor import ModuleFeatureExtractor
from optparse import OptionParser

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'


def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)

    parser.add_option("-t", "--target", dest="target", default="enrollment")
    parser.add_option("-d", "--dtype", dest="dtype", default="train")
    parser.add_option("-l", "--log", dest="log", default="log.csv")
    parser.add_option("-f", "--feature", dest="feature", default="feature.csv")
    parser.add_option("-m", "--mode", dest="mode", default="normal")
    parser.add_option("-n", "--limit", dest="limit", default="1000")

    parser.add_option("-e", "--epath", dest="epath",default="enrollment.csv")
    parser.add_option("-o", "--object", dest="object", default="object.csv")
    parser.add_option("-g", "--label", dest="label", default="label.csv")

    (options, remainder) = parser.parse_args()

    target = options.target
    data_type = options.dtype
    mode = options.mode
    debug_limit = int(options.limit)

    log_path = options.log
    feature_path = options.feature
    enrollment_path = options.epath
    object_path = options.object
    label_path = options.label

    extractor = None
    if target == 'enrollment':
        extractor = EnrollmentFeatureExtractor(mode, data_type, log_path, feature_path, debug_limit)
    elif target == 'course':
        extractor = CourseFeatureExtractor(mode, data_type, log_path, enrollment_path, label_path, object_path, feature_path, debug_limit)
    # elif target == 'module':
    #     extractor = ModuleFeatureExtractor(mode, data_type, log_path, object_path, feature_path, debug_limit)
    else:
        print 'Oops: The extractor %s is not supported now!' % target
        sys.exit(-1)
    extractor.extract()

if __name__ == '__main__':
    main()
