nproc_per_node=8

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=8

# nohup bash scripts/rm.sh > qwen_tool_all_data_180k_3epoch_4096_all_2round_maskstep1_code.log 2>&1 &
bsz=2
#501760
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct/" 
output_dir="."
DATASET_1="data/jsonl/wo_thinking_thyme_single_round.jsonl"
DATASET_2="data/jsonl/2round.jsonl"

export WANDB_PROJECT="codev-sft"
export WANDB_RUN_NAME="sft_stage1_$(date +%Y%m%d_%H%M%S)"

FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=3211264 \
swift sft \
    --model $MODEL_PATH \
    --dataset $DATASET_1 \
    	      $DATASET_2 \
    --train_type full \
    --lora_rank 8 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --system scripts/prompt.txt \
    --num_train_epochs 3 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $bsz) \
    --save_strategy steps \
    --save_steps 500 \
    --max_length 10240 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --save_safetensors true \
    --report_to wandb
