#!/bin/bash

if [ "$#" -ne 7 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 7 parameters for the GPUs, way, shot, data, strategy, hop and seed"
  exit 1 
fi
gpus=$1
arch=simpleCnn
att_arch=Attention_cosine
lr=1e-3
img_up_bound=10
batch_size=128
weight_decay=1e-5
classes_per_it_tr=$2
num_support_tr=$3
num_query_tr=$3
classes_per_it_val=$2
num_support_val=$3
num_query_val=$3
data=$4
training_strategy=$5
n_hop=$6
seed=$7
reset_interval=5

CUDA_VISIBLE_DEVICES=${gpus} python ./exp/test_all_settings.py \
	       --manual_seed ${seed} \
	       --reset_interval ${reset_interval} \
	       --coef_base 1 --coef_anc 1 --coef_chi 0 --training_strategy ${training_strategy} \
               --arch ${arch} --att_arch ${att_arch} --n_hop ${n_hop} \
	       --dataset_root ${HOME}/datasets/tiered-imagenet-${data}/ --workers 16 \
	       --log_dir ./logs.buffer/TEST-${data}_${classes_per_it_val}way${num_support_val}shot_base1_anc1_chi0_ppbuffer_${training_strategy}_nhop${n_hop}_reset${reset_interval}/ --log_interval 20 --test_interval 100 \
	       --epochs 1500 --start_decay_epoch 100 --iterations 100 --batch_size ${batch_size} --img_up_bound ${img_up_bound} --lr ${lr} --lr_step 15000 --lr_gamma 0.7 --weight_decay ${weight_decay} \
	       --classes_per_it_tr ${classes_per_it_tr} --num_support_tr ${num_support_tr} --num_query_tr ${num_query_tr} --classes_per_it_val ${classes_per_it_val} --num_support_val ${num_support_val}  --num_query_val ${num_query_val}

