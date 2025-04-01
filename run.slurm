#!/bin/bash

#SBATCH --job-name=distilltruth
#SBATCH --partition=cuda-gpu
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_logs/distilltruth_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jebeene@purdue.edu
#
eval "$(/homes/jebeene/scratch/miniconda3/bin/conda shell.bash hook)"
conda activate distill-truth

python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python3 main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

conda deactivate
