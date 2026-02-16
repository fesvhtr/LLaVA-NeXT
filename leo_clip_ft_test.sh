#!/bin/bash
#SBATCH --job-name=llava_leo_clip_r_336_s1_ft_test
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=llava_leo_clip_r_336_s1_ft_test.out
#SBATCH --error=llava_leo_clip_r_336_s1_ft_test.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
set -euo pipefail

export HF_HOME="/leonardo_work/EUHPC_R04_192/fmohamma/zsc/hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load profile/deeplrn
module load openmpi
module load cuda/11.8
source $WORK/fmohamma/venvs/llava_zsc/bin/activate
cd $WORK/fmohamma/zsc/LLaVA-NeXT

LLM_VERSION="/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="Qwen2-7B-Instruct"
VISION_MODEL_VERSION="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_s1/run_1215_081150/finetune_weights/checkpoint-1280"
VISION_MODEL_VERSION_CLEAN="clip_r_336_s1"
VISION_TOWER_PROCESSOR="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"

############### Finetune ################
PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-ft-llava_1_6"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

MASTER_ADDR="localhost"
MASTER_PORT=29500
NUM_WORKERS=8

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] NUM_WORKERS(per process)=$NUM_WORKERS"

LAUNCH_CMD="accelerate launch \
    --mixed_precision=bf16 \
    --num_machines 1 \
    --num_processes 1 \
    --machine_rank 0 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path ${LLM_VERSION} \
        --version ${PROMPT_VERSION} \
        --data_path=/leonardo_scratch/large/userexternal/fmohamma/zsc/llava_data/llava_1_6.json \
        --image_folder /leonardo_scratch/large/userexternal/fmohamma/zsc/llava_data/llava_1_6_images \
        --pretrain_mm_mlp_adapter=\"/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin\" \
        --mm_tunable_parts=\"mm_vision_tower,mm_mlp_adapter,mm_language_model\" \
        --mm_vision_tower_lr=2e-6 \
        --vision_tower ${VISION_MODEL_VERSION} \
        --vision_tower_processor ${VISION_TOWER_PROCESSOR} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --group_by_modality_length True \
        --image_aspect_ratio anyres \
        --image_grid_pinpoints \"[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]\" \
        --mm_patch_merge_type spatial_unpad \
        --bf16 True \
        --run_name $MID_RUN_NAME \
        --output_dir /checkpoints/${MID_RUN_NAME} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 3000 \
        --save_total_limit 1 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 32768 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --lazy_preprocess True \
        --report_to wandb \
        --torch_compile True \
        --torch_compile_backend inductor \
        --dataloader_drop_last True \
        --attn_implementation sdpa"

srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 \
    bash -c "$LAUNCH_CMD"

echo "LLaVA-NeXT (single-node single-gpu) clip ft completed."
