#!/bin/bash
###############################################################################
# run_thorough.sh — Complete VLN training pipeline (Tasks 2-5)
#
# Produces all results in results/ with images, videos, graphs, and
# a comprehensive markdown report.
#
# Expected runtime: ~1.5-2 hours on RTX 3050 6GB
# Usage:  chmod +x run_thorough.sh && nohup ./run_thorough.sh > thorough.log 2>&1 &
###############################################################################

# ---------- environment setup ----------
export TMPDIR="${HOME}/habitat/.tmp"
mkdir -p "$TMPDIR"

PYTHON="/home/hans/miniconda3/envs/habitat/bin/python"
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

LOG="results/thorough_run.log"
mkdir -p results/{task2,task3,task4,task5}

# Activate conda (for scripts that need it)
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate habitat 2>/dev/null || true

# Verify python
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Falling back to conda env python..."
    PYTHON="python"
fi

# Logging helper
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

# ---------- start ----------
log "========================================================="
log "THOROUGH VLN TRAINING PIPELINE — START"
log "========================================================="
log "Working directory: $REPO"
log "Python: $PYTHON"
log "Log file: $LOG"
log "TMPDIR: $TMPDIR"

$PYTHON -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
" 2>&1 | tee -a "$LOG"

# Clean old checkpoints to save disk space (models will be retrained)
log "Cleaning old checkpoints to free disk space..."
find "results/task3/checkpoints/" -name "*.pt" -delete 2>/dev/null
find "results/task5/" -name "*.pt" -delete 2>/dev/null
find "results/task2/" -name "*.pt" -delete 2>/dev/null
rm -rf outputs/ 2>/dev/null
log "Disk space: $(df -h / | tail -1 | awk '{print $4}') available"

OVERALL_START=$(date +%s)
TASK_STATUS=""

###############################################################################
# TASK 2: Baseline VLN Training
###############################################################################
log ""
log "========================================================="
log "TASK 2: Baseline VLN Training (15 epochs)"
log "========================================================="
TASK2_START=$(date +%s)

if $PYTHON task2_vln/train.py \
    --train-jsonl "task 3/data_large/train/samples.jsonl" \
    --val-jsonl "task 3/data_large/val/samples.jsonl" \
    --output-dir "results/task2" \
    --epochs 15 \
    --batch-size 8 \
    --lr 2e-4 \
    --label-smoothing 0.1 \
    --device auto \
    --augment \
    2>&1 | tee -a "$LOG"; then
    TASK2_TIME=$(( $(date +%s) - TASK2_START ))
    log "✅ Task 2 completed in $(( TASK2_TIME / 60 ))m $(( TASK2_TIME % 60 ))s"
    TASK_STATUS="${TASK_STATUS}Task 2: ✅ PASS (${TASK2_TIME}s)\n"
else
    TASK2_TIME=$(( $(date +%s) - TASK2_START ))
    log "❌ Task 2 failed after $(( TASK2_TIME / 60 ))m $(( TASK2_TIME % 60 ))s"
    TASK_STATUS="${TASK_STATUS}Task 2: ❌ FAIL\n"
fi

###############################################################################
# TASK 3: Hyperparameter Sweep + SR/SPL Evaluation
###############################################################################
log ""
log "========================================================="
log "TASK 3: Hyperparameter Sweep & Evaluation (15 epochs × 4 configs)"
log "========================================================="
TASK3_START=$(date +%s)

if $PYTHON task2_vln/scripts/task3_train_eval.py \
    --data-root "task 3/data_large" \
    --artifacts-root "results/task3" \
    --epochs 15 \
    --device auto \
    --augment \
    --sweep-lrs "1e-4,2e-4" \
    --sweep-batch-sizes "8,16" \
    --label-smoothing 0.1 \
    --num-workers 4 \
    2>&1 | tee -a "$LOG"; then
    TASK3_TIME=$(( $(date +%s) - TASK3_START ))
    log "✅ Task 3 completed in $(( TASK3_TIME / 60 ))m $(( TASK3_TIME % 60 ))s"
    TASK_STATUS="${TASK_STATUS}Task 3: ✅ PASS (${TASK3_TIME}s)\n"
else
    TASK3_TIME=$(( $(date +%s) - TASK3_START ))
    log "⚠️  Task 3 may have partially failed (training likely succeeded, rollout may have failed)"
    log "    Elapsed: $(( TASK3_TIME / 60 ))m $(( TASK3_TIME % 60 ))s"
    TASK_STATUS="${TASK_STATUS}Task 3: ⚠️  PARTIAL (${TASK3_TIME}s)\n"
fi

# Free disk: remove last.pt duplicates (only best.pt needed)
log "Cleaning last.pt files to save disk..."
find "results/task3/checkpoints/" -name "last.pt" -delete 2>/dev/null
find "results/task2/" -name "last.pt" -delete 2>/dev/null
log "Disk space: $(df -h / | tail -1 | awk '{print $4}') available"

###############################################################################
# Find best Task 3 checkpoint for downstream tasks
###############################################################################
BEST_CKPT=""
if [ -f "results/task3/logs/hyperparam_results.json" ]; then
    BEST_RUN=$($PYTHON -c "
import json
with open('results/task3/logs/hyperparam_results.json') as f:
    results = json.load(f)
if results:
    print(results[0]['run_name'])
" 2>/dev/null)
    if [ -n "$BEST_RUN" ] && [ -f "results/task3/checkpoints/${BEST_RUN}/best.pt" ]; then
        BEST_CKPT="results/task3/checkpoints/${BEST_RUN}/best.pt"
        log "Best checkpoint: $BEST_CKPT (run: $BEST_RUN)"
    fi
fi

# Fallback: find any best.pt
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT=$(find "results/task3/checkpoints/" -name "best.pt" 2>/dev/null | head -1)
    if [ -n "$BEST_CKPT" ]; then
        log "Fallback checkpoint: $BEST_CKPT"
    else
        log "WARNING: No checkpoint found! Task 4 generalization will use defaults."
        BEST_CKPT="results/task3/checkpoints/lr_0p0001_bs_8/best.pt"
    fi
fi

###############################################################################
# TASK 4a: Ablation Study
###############################################################################
log ""
log "========================================================="
log "TASK 4a: Ablation Study (4 configs × 10 epochs)"
log "========================================================="
TASK4A_START=$(date +%s)

if $PYTHON "task 4/scripts/task4_ablation_study.py" \
    --train-data "task 3/data_large/train/samples.jsonl" \
    --val-data "task 3/data_large/val/samples.jsonl" \
    --out-dir "results/task4" \
    --epochs 10 \
    --batch-size 8 \
    --lr 2e-4 \
    --device auto \
    2>&1 | tee -a "$LOG"; then
    TASK4A_TIME=$(( $(date +%s) - TASK4A_START ))
    log "✅ Task 4a (Ablation) completed in $(( TASK4A_TIME / 60 ))m $(( TASK4A_TIME % 60 ))s"
    TASK_STATUS="${TASK_STATUS}Task 4a Ablation: ✅ PASS (${TASK4A_TIME}s)\n"
else
    TASK4A_TIME=$(( $(date +%s) - TASK4A_START ))
    log "❌ Task 4a (Ablation) failed after $(( TASK4A_TIME / 60 ))m"
    TASK_STATUS="${TASK_STATUS}Task 4a Ablation: ❌ FAIL\n"
fi

###############################################################################
# TASK 4b: Generalization Evaluation
###############################################################################
log ""
log "========================================================="
log "TASK 4b: Generalization Evaluation"
log "========================================================="
TASK4B_START=$(date +%s)

if $PYTHON "task 4/scripts/task4_evaluate_generalization.py" \
    --out-dir "results/task4" \
    --device auto \
    --checkpoint "$BEST_CKPT" \
    --data-root "task 4/data" \
    --task3-root "task 3/data_large" \
    2>&1 | tee -a "$LOG"; then
    TASK4B_TIME=$(( $(date +%s) - TASK4B_START ))
    log "✅ Task 4b (Generalization) completed in $(( TASK4B_TIME / 60 ))m $(( TASK4B_TIME % 60 ))s"
    TASK_STATUS="${TASK_STATUS}Task 4b Generalization: ✅ PASS (${TASK4B_TIME}s)\n"
else
    TASK4B_TIME=$(( $(date +%s) - TASK4B_START ))
    log "❌ Task 4b (Generalization) failed after $(( TASK4B_TIME / 60 ))m"
    TASK_STATUS="${TASK_STATUS}Task 4b Generalization: ❌ FAIL\n"
fi

# Free disk space before Task 5
log "Cleaning last.pt from task4 artifacts..."
find "results/task4/" -name "last.pt" -delete 2>/dev/null
log "Disk space: $(df -h / | tail -1 | awk '{print $4}') available"

###############################################################################
# TASK 5: Controlled Extension (Baseline vs Lightweight)
###############################################################################
log ""
log "========================================================="
log "TASK 5: Controlled Extension (2 configs × 12 epochs)"
log "========================================================="
TASK5_START=$(date +%s)

if $PYTHON "task 5/scripts/task5_controlled_extension.py" \
    --data-root "task 3/data_large" \
    --output-dir "results/task5" \
    --epochs 12 \
    --batch-size 8 \
    --lr 1e-4 \
    --device auto \
    2>&1 | tee -a "$LOG"; then
    TASK5_TIME=$(( $(date +%s) - TASK5_START ))
    log "✅ Task 5 completed in $(( TASK5_TIME / 60 ))m $(( TASK5_TIME % 60 ))s"
    TASK_STATUS="${TASK_STATUS}Task 5: ✅ PASS (${TASK5_TIME}s)\n"
else
    TASK5_TIME=$(( $(date +%s) - TASK5_START ))
    log "❌ Task 5 failed after $(( TASK5_TIME / 60 ))m"
    TASK_STATUS="${TASK_STATUS}Task 5: ❌ FAIL\n"
fi

###############################################################################
# GENERATE FINAL REPORT
###############################################################################
log ""
log "========================================================="
log "Generating Final Report & Collecting Artifacts"
log "========================================================="

if $PYTHON scripts/generate_final_report.py 2>&1 | tee -a "$LOG"; then
    log "✅ Final report generated"
else
    log "⚠️  Report generation had issues (partial results may be available)"
fi

###############################################################################
# SUMMARY
###############################################################################
OVERALL_TIME=$(( $(date +%s) - OVERALL_START ))
log ""
log "========================================================="
log "PIPELINE COMPLETE"
log "========================================================="
log "Total time: $(( OVERALL_TIME / 3600 ))h $(( (OVERALL_TIME % 3600) / 60 ))m $(( OVERALL_TIME % 60 ))s"
log ""
log "Task Status:"
echo -e "$TASK_STATUS" | tee -a "$LOG"
log ""
log "Results directory contents:"
find results/ -type f | sort | tee -a "$LOG"
log ""
log "Disk usage:"
du -sh results/ | tee -a "$LOG"
log ""
log "========================================================="
log "📋 Final Report: results/final_report.md"
log "📊 Dashboard:    results/summary_dashboard.png"
log "📁 All results:  results/"
log "========================================================="
