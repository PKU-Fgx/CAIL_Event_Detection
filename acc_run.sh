accelerate launch main.py \
    --eva_step 50 \
    --do_train \
    --do_valid \
    --max_seq_length 512 \
    --model_name "bert-base-cn-law" \
    --save_name "bert-base-cn-law" \
    --train_batch_size 16 \
    --valid_batch_size 32 \
    --max_grad_norm 10.0 \
    --warmup_type "linear" \
    --lr 5e-5 \
    --num_train_epochs 10 \
    --eval_begin_epoch 1 \
    --seed 100 \
    --use_mutil_gpu \
    # --use_crf \