#!/bin/sh

gpu_id=2

fold_n=2

python main.py \
    --do_train \
    --do_valid \
    --do_save \
    --do_save_data \
    --data_fold "/MutilFold/fold_${fold_n}/" \
    --save_name "bert-CRF-Fold[${fold_n}]-3Max" \
    --gpu_idx $gpu_id \
    --eva_step 150 \
    --model_name "bert-base-chinese" \
    --train_batch_size 16 \
    --warmup_type "linear" \
    --lr 1e-4 \
    --use_crf \
    --use_focal \
    --focal_weight 0.8 \
    --focal_begin_epoch 2 \
    --fp16 \
    --last_n 3 \
    --pooling_type "max" \

python infer.py \
    --gpu_idx $gpu_id \
    --model_saved "bert-CRF-Fold[${fold_n}]-3Max-Stage2[]-Dev[]"
