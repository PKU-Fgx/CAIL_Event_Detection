#!/bin/sh

n=0

gpu_id=$n
fold_n=2

python infer.py \
    --gpu_idx $gpu_id \
    --model_saved "bert-CRF-Fold[${fold_n}]-3Sum-Stage2[]-Dev[]" \
    --do_save_data \