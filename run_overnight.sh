#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting overnight training pipeline across all tasks..."

# Activate your conda environment (adjust the hook if necessary)
eval "$(conda shell.bash hook)"
conda activate habitat

echo "========================================================="
echo "Running Task 2: Baseline VLN Training"
echo "========================================================="
python task2_vln/train.py --train-jsonl "task 3/data_large/train/samples.jsonl" --val-jsonl "task 3/data_large/val/samples.jsonl" --output-dir "outputs/task2_clip_crossattn" --epochs 10

echo "========================================================="
echo "Running Task 3: Large-Scale GPU Training and Evaluation"
echo "========================================================="
python task2_vln/scripts/task3_train_eval.py --data-root "task 3/data_large" --artifacts-root "task 3/artifacts_large" --epochs 10 --device auto --augment

echo "========================================================="
echo "Running Task 4: Ablation Study"
echo "========================================================="
python "task 4/scripts/task4_ablation_study.py" --train-data "task 3/data_large/train/samples.jsonl" --val-data "task 3/data_large/val/samples.jsonl" --out-dir "task 4/artifacts" --epochs 10

echo "========================================================="
echo "Running Task 4: Generalization Evaluation"
echo "========================================================="
python "task 4/scripts/task4_evaluate_generalization.py" --out-dir "task 4/artifacts" --device auto

echo "========================================================="
echo "Running Task 5: Controlled Extension (Lightweight Variant)"
echo "========================================================="
python "task 5/scripts/task5_controlled_extension.py" --data-root "task 3/data_large" --output-dir "task 5/artifacts" --epochs 10

echo "========================================================="
echo "✅ All tasks completed successfully! The graphs, models, and videos are securely saved."
echo "========================================================="