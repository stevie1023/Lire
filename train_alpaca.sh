export MODEL_PATH="/private/model/meta-llama/alpaca-7b"
export SAVE_PATH=$1
export DATA_PATH="/private/home/zhumingye/code/LIRE_acl24/data_generation/data/alpaca_responses_hh.json" ###hh-rlhf data path
export MASTER_ADDR="localhost"
export MASTER_PORT="23385"
export WANDB_DISABLED=true
wandb offline

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3.10 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=4 --use_env train_alpaca_prompt.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 200 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_total_limit 3 \
    --deepspeed "./default_offload_opt_param.json"\
    --resume_from_checkpoint "/data/rmw/RRHF/results/alpaca-mrt-old-hh-2e/checkpoint-1190"\
    --tf32 True --model_max_length 512 --lire_weight 1.0 --train_sample_num 6  ######set the sample num to 2 if DPO loss is applied 
