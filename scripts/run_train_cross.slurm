#!/bin/bash
#SBATCH --job-name=eeg_train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --output=./logs/train_%j.out
#SBATCH --error=./logs/train_%j.err



# Notifications par email
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=embiteye@etu.uqac.ca

module load python/3.10
module load cuda/12.2

# Active ton environnement
source ~/EEG_Emotion_Project/venv/bin/activate

echo "=== Lancement main_train.py ==="
python -u ~/EEG_Emotion_Project/code/main_train_cross.py

echo "=== Fin du job ==="


