# !/bin/bash

#########################################################
#     This script performs the following operations:    #
#     1. Downloads the ImageNet dataset                 #
#     2. Train an Alexnet model on the training set     #
#     3. Evaluate the model on the validation set       #
#########################################################


# Download the dataset
#CUDA_VISIBLE_DEVICES=0 python convert_data.py --flagfile=conf/convert.conf

CUDA_VISIBLE_DEVICES=0 python convert_data.py \
  --dataset_dir='/data/donghao1/project/imagenet/images' \
  --train_label_dir='/data/donghao1/project/imagenet/tool/train.txt' \
  --val_label_dir='/data/donghao1/project/imagenet/tool/val.txt'

# Train model
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
  --dataset_dir='/data/donghao1/project/imagenet/images' \

# Evaluate model
#CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py --conf=conf/eval.conf

