#!/bin/bash
# Task 3: Full GPU Training Pipeline
# This script: 1) waits for GPU drivers, 2) generates expanded dataset, 3) trains on GPU

set -e

HABITAT_ROOT="/home/hans/habitat"
DATA_ROOT="${HABITAT_ROOT}/task 3/data_large"
ARTIFACTS_ROOT="${HABITAT_ROOT}/task 3/artifacts_large"
CONDA_ENV="habitat"

echo "================================"
echo "Task 3: GPU Training Pipeline"
echo "================================"

# Step 1: Wait for NVIDIA drivers to finish installing
echo ""
echo "[Step 1/4] Waiting for NVIDIA drivers to install..."
timeout 1800 bash -c 'while pgrep -f "nvidia-driver" > /dev/null 2>&1; do sleep 10; done' || true

# Give drivers a moment to fully load
sleep 5

# Step 2: Verify GPU
echo "[Step 2/4] Verifying GPU availability..."
source /home/hans/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

python - <<'PY'
import torch
print("\n=== GPU Status ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
else:
    print("⚠ WARNING: CUDA not available - will train on CPU (slow)")
print()
PY

# Step 3: Generate large dataset if not exists
echo "[Step 3/4] Generating large-scale dataset (500 train, 100 val)..."
if [ ! -d "${DATA_ROOT}" ]; then
    cd "${HABITAT_ROOT}"
    python task2_vln/scripts/task3_generate_data_large.py \
        --out-dir "${DATA_ROOT}" \
        --train-episodes 500 \
        --val-episodes 100
else
    echo "  Dataset already exists at ${DATA_ROOT}"
fi

# Step 4: Train model with extended epochs and hyperparameters
echo "[Step 4/4] Training model with hyperparameter sweep..."
echo "  Output directory: ${ARTIFACTS_ROOT}"
echo ""

cd "${HABITAT_ROOT}"
python task2_vln/scripts/task3_train_eval.py \
    --data-root "${DATA_ROOT}" \
    --artifacts-root "${ARTIFACTS_ROOT}" \
    --epochs 5 \
    --device auto \
    --seed 42

echo ""
echo "✓ Training complete! Results saved to:"
echo "  Artifacts: ${ARTIFACTS_ROOT}"
echo ""
echo "Key outputs:"
echo "  - Learning curves: ${ARTIFACTS_ROOT}/checkpoints/*/learning_curves.png"
echo "  - Confusion matrices: ${ARTIFACTS_ROOT}/plots/confusion_*.png"
echo "  - SR/SPL metrics: ${ARTIFACTS_ROOT}/logs/task3_metrics.json"
echo "  - Rollout videos: ${ARTIFACTS_ROOT}/videos/rollout_*.gif"
echo ""
