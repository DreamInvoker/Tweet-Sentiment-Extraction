#! /bin/bash

export CUDA_VISIBLE_DEVICES=$1
nohup python3.6 -u main.py \
--model roberta-base \
--num_warmup_steps 100 \
> train.log 2>&1 &

#--model roberta-large \
