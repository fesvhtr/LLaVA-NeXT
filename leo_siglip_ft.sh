#!/bin/bash
#SBATCH --job-name=llava_leo_siglip_r_s2_ft
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=llava_leo_siglip_r_s2_ft.out
#SBATCH --error=llava_leo_siglip_r_s2_ft.err
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

############### Finetune ################
PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-ft-llava_1_6"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29000 + SLURM_JOBID % 1000))
NUM_WORKERS=8

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] NUM_WORKERS(per process)=$NUM_WORKERS"

LAUNCH_CMD=""
srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=32 \
    bash -c "$LAUNCH_CMD"

echo "LLaVA-NeXT (multi-node) siglip ft completed."
# You can delete the sdpa attn_implementation if you want to use flash attn
