#!/usr/bin/env bash

root=$1
dataset=$2

# cluster
python extractor.py --arch moco --dataset $dataset --data_root $root --batch_size 32 --mode cluster --sample_num=20000 --k 50

# assign 
python extractor.py --arch moco --dataset $dataset --data_root $root --batch_size 32 --mode assign --k 50