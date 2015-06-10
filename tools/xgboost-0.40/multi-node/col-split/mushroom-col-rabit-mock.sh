#!/bin/bash
if [[ $# -ne 1 ]]
then
    echo "Usage: nprocess"
    exit -1
fi

#
# This script is same as mushroom-col except that we will be using xgboost instead of xgboost-mpi
# xgboost used built in tcp-based allreduce module, and can be run on more enviroment, so long as we know how to start job by modifying ../submit_job_tcp.py
#
rm -rf train.col* *.model
k=$1

# split the lib svm file into k subfiles
python splitsvm.py ../../demo/data/agaricus.txt.train train $k

# run xgboost mpi
../../subtree/rabit/tracker/rabit_demo.py -n $k  ../../xgboost.mock mushroom-col.conf dsplit=col mock=0,2,0,0 mock=1,2,0,0 mock=2,2,8,0 mock=2,3,0,0

# the model can be directly loaded by single machine xgboost solver, as usuall
#../../xgboost mushroom-col.conf task=dump model_in=0002.model fmap=../../demo/data/featmap.txt name_dump=dump.nice.$k.txt


#cat dump.nice.$k.txt
