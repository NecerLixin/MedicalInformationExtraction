#!/bin/bash

# 设置默认参数值
lr=5e-5
epochs=20
batch_size=24
num_labels=331
device="cpu"
save_path="model_save/model_cls_bert_tree.pth"
pretrained_model="bert-base-chinese"
warmup_rate=0.06
train_data_path="nlp2024-data/dataset/small_train.json"
dev_data_path="nlp2024-data/dataset/small_dev.json"
test_data_path="nlp2024-data/dataset/small_dev.json"
info="CLS BERT with syntax tree."

# 运行Python脚本并传递参数
python3 train_bert_tree.py \
    --lr $lr \
    --epochs $epochs \
    --batch_size $batch_size \
    --num_labels $num_labels \
    --device $device \
    --save_path $save_path \
    --pretrained_model $pretrained_model \
    --warmup_rate $warmup_rate \
    --train_data_path $train_data_path \
    --dev_data_path $dev_data_path \
    --test_data_path $test_data_path \
    --info "$info"
