#!/bin/bash
#SBATCH --job-name=llava_leo_siglip_s2_pretrain
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=llava_leo_siglip_s2_pretrain.out
#SBATCH --error=llava_leo_siglip_s2_pretrain.err
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
VISION_MODEL_VERSION="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_s2/run_0203_195004/finetune_weights/checkpoint-673"
VISION_MODEL_VERSION_CLEAN="siglip_r_s2"
VISION_TOWER_PROCESSOR="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29000 + SLURM_JOBID % 1000))
NUM_WORKERS=8

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] NUM_WORKERS(per process)=$NUM_WORKERS"

LAUNCH_CMD="accelerate launch \
  --multi_gpu \
  --mixed_precision=bf16 \
  --num_machines 2 \
  --num_processes 8 \
  --machine_rank \$SLURM_NODEID \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /leonardo_scratch/large/userexternal/fmohamma/zsc/llava_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /leonardo_scratch/large/userexternal/fmohamma/zsc/llava_data/LLaVA-Pretrain \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_tower_processor ${VISION_TOWER_PROCESSOR} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $WORK/fmohamma/zsc/LLaVA-NeXT/checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa"

srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=32 \
    bash -c "$LAUNCH_CMD"

echo "LLaVA-NeXT (multi-node) siglip pretrain completed."
# You can delete the sdpa attn_implementation if you want to use flash attn