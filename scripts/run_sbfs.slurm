#!/bin/bash
#SBATCH --job-name=sbfs_EEG
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=./logs/sbfs_%j.out
#SBATCH --error=./logs/sbfs_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=embiteye@etu.uqac.ca

module load python/3.10
module load cuda/12.2

source ~/EEG_Emotion_Project/venv/bin/activate

cd ~/EEG_Emotion_Project

echo "=== Lancement main_selection_sbfs.py ==="
python -u ~/EEG_Emotion_Project/code/selection/main_selection_sbfs.py
echo "=== Job terminé ==="
