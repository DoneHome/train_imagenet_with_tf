# !/bin/bash

#########################################################
#     This script performs the following operations:    #
#     1. Downloads the ImageNet dataset                 #
#     2. Train an Alexnet model on the training set     #
#     3. Evaluate the model on the validation set       #
#########################################################


# Download the dataset
#CUDA_VISIBLE_DEVICES=0 python convert_data.py --flagfile=conf/convert.conf

CUDA_VISIBLE_DEVICES=3 python convert_data.py \
  --dataset_dir='/data/donghao1/project/imagenet/images' \
  --train_label_dir='/data/donghao1/project/imagenet/tool/train.txt' \
  --val_label_dir='/data/donghao1/project/imagenet/tool/val.txt'

# Train model
CUDA_VISIBLE_DEVICES=3 python train_image_classifier.py \
  --dataset_dir='/data/donghao1/project/imagenet/images' \
  --train_dir='/data/donghao1/project/train_imagenet_with_tf/log' \
  --num_epochs=1 \
  --batch_size=128 \
  --learning_rate=0.001 \
  --weight_decay=0.00004 \
  --log_every_n_steps=10 \
  --save_model_steps=5000 \




# Evaluate model
#CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py --conf=conf/eval.conf

