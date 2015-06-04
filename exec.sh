# args
mode=''
debug_limit=''
if [ $# == 2 ]; then
    mode=$1
    debug_limit=$2
fi

python py/extract_feature.py enrollment train data/log_train.csv data/train_enrollment_feature.csv $mode $debug_limit
python py/extract_feature.py enrollment test data/log_test.csv data/test_enrollment_feature.csv $mode $debug_limit