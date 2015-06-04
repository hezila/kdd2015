# args
mode=''
debug_limit=''
if [ $# == 2 ]; then
    mode=$1
    debug_limit=$2
fi

python python/extract_feature.py enrollment train data/log_train.csv data/train_feature.csv $mode $debug_limit
