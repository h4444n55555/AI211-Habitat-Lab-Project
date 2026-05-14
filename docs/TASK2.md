# Task 2 Implementation Status (Completed)

This file summarizes what was implemented for **Task 2: Vision-Language Model Implementation** and where everything is located.

## What Was Implemented

### 1) Pretrained Visual Encoder
- Implemented with CLIP ViT-B/16 vision backbone.
- Source: Hugging Face `openai/clip-vit-base-patch16`.

### 2) Pretrained Text Encoder
- Implemented with CLIP text transformer from the same pretrained checkpoint.
- Keeps text and visual embeddings in a shared representation space.

### 3) Multimodal Fusion Module
- Implemented cross-attention fusion:
  - text tokens are query
  - image tokens are key/value
- Includes residual connection + layer norm.

### 4) Policy Head
- 4-way action classifier implemented:
  - `0: move_forward`
  - `1: turn_left`
  - `2: turn_right`
  - `3: stop`

### 5) Training and Validation Pipeline
- Full train loop with:
  - dataloading
  - cross-entropy loss
  - optimizer step
  - gradient clipping
  - checkpoint saving (`best.pt`, `last.pt`)
- Validation loop with accuracy metric.
- Learning curves plot output (`learning_curves.png`).

## File Map

- Main package: `/home/hans/habitat/task2_vln`
- Model: `/home/hans/habitat/task2_vln/models/vln_policy.py`
- Dataset loader: `/home/hans/habitat/task2_vln/data/dataset.py`
- Train script: `/home/hans/habitat/task2_vln/train.py`
- Eval script: `/home/hans/habitat/task2_vln/eval.py`
- Dummy data generator: `/home/hans/habitat/task2_vln/scripts/make_dummy_data.py`
- Default config: `/home/hans/habitat/task2_vln/configs/train_config.yaml`
- Dependency list: `/home/hans/habitat/task2_vln/requirements.txt`
- Task 2 usage guide: `/home/hans/habitat/task2_vln/README.md`

## How To Run

### 1) Activate environment

```bash
source /home/hans/miniconda3/etc/profile.d/conda.sh
conda activate habitat
```

### 2) Install dependencies

```bash
pip install -r /home/hans/habitat/task2_vln/requirements.txt
```

### 3) Create dummy dataset (sanity test)

```bash
python /home/hans/habitat/task2_vln/scripts/make_dummy_data.py --out-dir /home/hans/habitat/task2_vln/dummy_data
```

### 4) Train

```bash
cd /home/hans/habitat/task2_vln
python train.py \
  --train-jsonl dummy_data/train.jsonl \
  --val-jsonl dummy_data/val.jsonl \
  --image-root dummy_data \
  --output-dir outputs/clip_crossattn \
  --epochs 5
```

### 5) Evaluate

```bash
cd /home/hans/habitat/task2_vln
python eval.py \
  --checkpoint outputs/clip_crossattn/best.pt \
  --eval-jsonl dummy_data/val.jsonl \
  --image-root dummy_data \
  --out-json outputs/clip_crossattn/eval_metrics.json
```

## Expected Real-Data Integration

To move from Task 2 to Task 3 experiments:
- replace dummy JSONL with real imitation-learning trajectories
- keep the same train/eval scripts
- report SR/SPL during rollouts in Habitat evaluation phase

## Notes

- Implementation is modular to support Task 4 ablations:
  - frozen vs partially fine-tuned CLIP
  - fusion replacements (MLP vs cross-attention)
- Current scripts are ready for immediate use in your existing `habitat` conda environment.
