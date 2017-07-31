# !/bin/bash

#########################################################
#     This script performs the following operations:    #
#     1. Downloads the ImageNet dataset                 #
#     2. Train an Alexnet model on the training set     #
#     3. Evaluate the model on the validation set       #
#########################################################


# Download the dataset

# Train model
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py --conf=conf/train.conf

# Evaluate model
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py --conf=conf/eval.conf

