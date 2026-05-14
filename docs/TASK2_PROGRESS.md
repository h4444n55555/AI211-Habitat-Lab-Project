# Task 2 Progress Report

## Status
Task 2 is implemented and smoke-tested.

## What Was Completed
- Built a CLIP-based vision-language navigation policy.
- Added a multimodal fusion block using cross-attention.
- Added a 4-way policy head for:
  - move_forward
  - turn_left
  - turn_right
  - stop
- Added train and eval scripts.
- Added a dataset loader for JSONL imitation-learning samples.
- Added a dummy-data generator for smoke testing.
- Added a root summary file: [TASK2.md](TASK2.md)

## Verified Outputs
The following artifacts were generated in the task folder:
- [task2_vln/outputs/run1/best.pt](task2_vln/outputs/run1/best.pt)
- [task2_vln/outputs/run1/last.pt](task2_vln/outputs/run1/last.pt)
- [task2_vln/outputs/run1/metrics.json](task2_vln/outputs/run1/metrics.json)
- [task2_vln/outputs/run1/eval_metrics.json](task2_vln/outputs/run1/eval_metrics.json)
- [task2_vln/outputs/run1/learning_curves.png](task2_vln/outputs/run1/learning_curves.png)

## Smoke-Test Results
- Training ran for 3 epochs on generated dummy data.
- Validation accuracy reached 1.0000.
- Evaluation accuracy was 1.0.
- Per-class accuracy was 1.0 for all four actions on the validation smoke test.

Evaluation metrics:
- accuracy: 1.0
- total_samples: 128

## Where the Code Lives
- Model: [task2_vln/models/vln_policy.py](task2_vln/models/vln_policy.py)
- Dataset loader: [task2_vln/data/dataset.py](task2_vln/data/dataset.py)
- Training script: [task2_vln/train.py](task2_vln/train.py)
- Evaluation script: [task2_vln/eval.py](task2_vln/eval.py)
- Dummy data generator: [task2_vln/scripts/make_dummy_data.py](task2_vln/scripts/make_dummy_data.py)
- Usage guide: [task2_vln/README.md](task2_vln/README.md)

## How To Reproduce the Output
```bash
cd /home/hans/habitat/task2_vln
source /home/hans/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python scripts/make_dummy_data.py --out-dir dummy_data --train-size 512 --val-size 128
python train.py \
  --train-jsonl dummy_data/train.jsonl \
  --val-jsonl dummy_data/val.jsonl \
  --image-root dummy_data \
  --output-dir outputs/run1 \
  --epochs 3 \
  --batch-size 16 \
  --num-workers 0 \
  --device cpu
python eval.py \
  --checkpoint outputs/run1/best.pt \
  --eval-jsonl dummy_data/val.jsonl \
  --image-root dummy_data \
  --batch-size 16 \
  --num-workers 0 \
  --device cpu \
  --out-json outputs/run1/eval_metrics.json
```

## Short TA-Facing Summary
Task 2 is complete at the code level and validated with a full smoke test. The pipeline includes a pretrained CLIP vision encoder, pretrained CLIP text encoder, cross-attention fusion, and a 4-action policy head, with train/eval scripts and saved outputs.
