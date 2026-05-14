# Task 2: Vision-Language Model Implementation

This folder contains a complete Task 2 pipeline for visual navigation action prediction.

## Model Stack

- Vision encoder: `openai/clip-vit-base-patch16` vision transformer
- Text encoder: `openai/clip-vit-base-patch16` text transformer
- Fusion: cross-attention (text queries attend to image tokens)
- Policy head: 4-way action classifier (`forward`, `left`, `right`, `stop`)

## Dataset Format

Training and validation files are JSONL with one record per sample:

```json
{"image": "images/train_00001.png", "instruction": "Turn left at the corridor.", "action": 1}
```

Fields:
- `image`: relative or absolute image path
- `instruction`: natural-language instruction string
- `action`: integer in `[0,3]`

## Quick Start

1. Activate env:

```bash
source /home/hans/miniconda3/etc/profile.d/conda.sh
conda activate habitat
```

2. Install Python dependencies:

```bash
pip install -r task2_vln/requirements.txt
```

3. Create dummy data for smoke tests:

```bash
python task2_vln/scripts/make_dummy_data.py --out-dir task2_vln/dummy_data
```

4. Train baseline:

```bash
cd /home/hans/habitat/task2_vln
python train.py \
  --train-jsonl dummy_data/train.jsonl \
  --val-jsonl dummy_data/val.jsonl \
  --image-root dummy_data \
  --output-dir outputs/clip_crossattn \
  --epochs 5
```

5. Evaluate checkpoint:

```bash
cd /home/hans/habitat/task2_vln
python eval.py \
  --checkpoint outputs/clip_crossattn/best.pt \
  --eval-jsonl dummy_data/val.jsonl \
  --image-root dummy_data \
  --out-json outputs/clip_crossattn/eval_metrics.json
```

## Notes for Assignment Task 2

- This implementation covers all required Task 2 components:
  - pretrained visual encoder
  - pretrained text encoder
  - multimodal fusion
  - policy head
  - training + validation pipeline
- For real results, replace dummy data JSONL with your imitation-learning trajectory dataset.
