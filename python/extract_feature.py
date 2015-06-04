#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse
import os, sys

from feature.enrollment_feature_extractor import EnrollmentFeatureExtractor

__author__ = 'Feng Wang (Felix)'
__email__ = 'wangfelix87@gmail.com'
__date__ = '06-04-2015'

# args
parser = argparse.ArgumentParser()
parser.add_argument('target', type=str, choices=['enrollment'], default='enrollment')
parser.add_argument('data_type', type=str, choices=['train', 'test'], default='train')
parser.add_argument('log_path', type=str, nargs='?', default='log.csv')
parser.add_argument('feature_path', type=str, nargs='?', default='feature.csv')
parser.add_argument('mode', type=str, choices=['debug', 'normal'], nargs='?', default='normal')
parser.add_argument('debug_limit', type=int, nargs='?', default=1000)

def main():
    args = parser.parse_args()
    target = args.target
    data_type = args.data_type
    mode = args.mode
    debug_limit = args.debug_limit

    log_path = args.log_path
    feature_path = args.feature_path

    extractor = None
    if target == 'enrollment':
        extractor = EnrollmentFeatureExtractor(mode, data_type, log_path, feature_path, debug_limit)
    
    extractor.extract()

if __name__ == '__main__':
    main()
