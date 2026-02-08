#!/bin/bash
#SBATCH --job-name=llava_next_data
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=llava_next_data.out
#SBATCH --error=llava_next_data.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=128G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# activate env
module load profile/deeplrn
module load openmpi
source "$WORK/fmohamma/venvs/llm/bin/activate"

cd "$WORK/fmohamma/LLaVA-NeXT"

python -u scripts/prepare_llava_next_data.py
