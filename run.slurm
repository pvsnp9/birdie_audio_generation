#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=birdie
#SBATCH --output=/projects/mzampier/birdie_audio_generation/src/outputs/%u/test.out
#SBATCH --error=/projects/mzampier/birdie_audio_generation/src/outputs/%u/test.err
#SBATCH --nodes=1
# SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem-per-cpu=20GB
#SBATCH --export=ALL
#SBATCH --time=2-00:00:00

set echo
umask 0027

# to see ID and state of GPUs assigned
# nvidia-smi

# module load gnu10                           
# module load python/3.9.9-jh
# module load cuda/12.4.0
# export PYTHONPATH=/projects/mzampier/python-packages/projectX:$PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate delta || { echo "Failed to activate the conda env (delta)"; exit 1; }


python runner.py || { echo "Python script failed"; exit 1; }

