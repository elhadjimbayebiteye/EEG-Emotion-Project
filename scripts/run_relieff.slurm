#!/bin/bash
#SBATCH --job-name=mrmr_train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --output=./logs/mrmr_%j.out
#SBATCH --error=./logs/mrmr_%j.err

# Notifications par email 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=embiteye@etu.uqac.ca

# Charger modules
module load python/3.10
module load cuda/12.2

# Active ton environnement
source ~/EEG_Emotion_Project/venv/bin/activate

echo "=== Lancement main_selection_relieff.py ==="

python -u ~/EEG_Emotion_Project/code/selection/main_selection_relieff.py

echo "=== Job terminé ==="
