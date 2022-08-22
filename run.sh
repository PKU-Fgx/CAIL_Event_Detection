rm /tf/FangGexiang/1.CAILED/ModelSaved/OfficalBaselineV1 -r
python baseline_bert_crf.py \
    --eva_step 1000 \
    --do_train \
    --do_valid \
    --max_seq_length 512 \
    --model_name "bert-base-chinese" \
    --train_batch_size 8 \
    --valid_batch_size 8 \
    --lr 5e-5 \
    --num_train_epochs 5 \
    --save_name "OfficalBaselineV1" \
    --gpu_idx 0 \
    --fp16 \
    # --overwrite_cache \
