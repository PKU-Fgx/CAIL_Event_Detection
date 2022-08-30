python main.py \
    --eva_step 200 \
    --do_train \
    --do_valid \
    --max_seq_length 512 \
    --model_name "nezha-base-wwm" \
    --save_name "nezha-base" \
    --train_batch_size 4 \
    --valid_batch_size 4 \
    --lr 5e-5 \
    --num_train_epochs 4 \
    --gpu_idx 0 \
    --fp16 \
