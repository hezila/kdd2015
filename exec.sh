# args
mode=''
debug_limit=''
if [ $# == 2 ]; then
    mode=$1
    debug_limit=$2
fi

pypy py/extract_feature.py -t enrollment -d train -l data/log_train.csv -f data/train_enrollment_feature.csv
pypy py/extract_feature.py -t enrollment -d test -l data/log_test.csv -f data/test_enrollment_feature.csv

pypy py/extract_feature.py -t course -d train -l data/log_train.csv -e data/enrollment_train.csv -g data/truth_train.csv -f data/train_course_feature.csv
pypy py/extract_feature.py -t course -d test -l data/log_test.csv -e data/enrollment_train.csv -g data/truth_train.csv -f data/test_course_feature.csv

# python py/extract_feature.py -t module -d train -l data/log_train.csv -o data/object.csv -f data/train_module_feature.csv
# python py/extract_feature.py -t module -d test -l data/log_test.csv -o data/object.csv -f data/test_module_feature.csv
