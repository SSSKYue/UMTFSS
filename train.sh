#!/bin/sh

dataset=$1
exp_name=$2

exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

if [ "$exp_name" = "ssl_resnet50" ] 
then 
    config=config/${dataset}/${dataset}_ssl_resnet50.yaml 
else 
    config=config/${dataset}/${dataset}_split0_resnet50.yaml 
fi

mkdir -p ${model_dir} 
mkdir -p ${result_dir}

now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${config} ${exp_dir}

python3 -u train.py --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
