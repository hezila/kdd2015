model='lgc'
paras='paras/lgc.json'
if [ $# == 2 ]; then
    model=$1
    paras=$2
fi

python main.py -t ../data/train_enrollment_feature.csv -l ../data/truth_train.csv -s ../data/test_enrollment_feature.csv -m $model -p $paras
