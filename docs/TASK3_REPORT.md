# Task 3: Vision-Language Navigation Training Report

Date: May 14, 2026  
Environment: NVIDIA GeForce RTX 3050 6GB (CUDA), Habitat-Sim MP3D scenes

## Summary
Task 3 was rerun on real Habitat-Sim rendered RGB trajectories generated from the local Matterport3D scene bundle. The previous synthetic-color shortcut dataset and inflated 100% claims are no longer used for this report.

Best sweep run (current pipeline):
- Run: `lr_0p0002_bs_8`
- Validation accuracy: 0.4599
- Validation loss: 1.5000

Navigation metrics on 20 validation episodes:
- SR: 0.10
- SPL: 0.10
- OSR: 0.25
- NE: 2.3922

## Dataset
Generated with `task2_vln/scripts/task3_generate_data_large.py`:
- Train episodes: 100
- Val episodes: 20
- Train samples: 2041
- Val samples: 411
- Source scenes: local MP3D/MP3D-example assets under `habitat-sim/data/scene_datasets`

Each episode contains:
- Habitat-rendered RGB frames
- Natural-language instruction derived from expert action plan
- Expert action sequence (`move_forward`, `turn_left`, `turn_right`, `stop`)
- Start and goal positions
- Scene path metadata for rollout evaluation

## Training Configuration
Script: `task2_vln/scripts/task3_train_eval.py`

Current run settings:
- Epochs: 1
- Sweep learning rates: `[1e-4, 2e-4]`
- Sweep batch sizes: `[8]`
- Device: auto (CUDA selected)
- CLIP backbone: `openai/clip-vit-base-patch16`
- Weighted CE + label smoothing
- Mixed precision (CUDA AMP)
- Cosine LR schedule

## Hyperparameter Sweep Results
From `task 3/artifacts_large/logs/hyperparam_results.json`:

1. `lr_0p0002_bs_8`
- best_val_acc: 0.45985401459854014
- final_val_loss: 1.500003378176631

2. `lr_0p0001_bs_8`
- best_val_acc: 0.3260340632603406
- final_val_loss: 1.508815393830738

## Rollout Evaluation
From `task 3/artifacts_large/logs/task3_metrics.json`:
- SR: 0.1
- SPL: 0.1
- OSR: 0.25
- NE: 2.392205610871315
- eval episodes: 20

Interpretation:
- The model is no longer trivially memorizing synthetic frame labels.
- Real-scene learning is substantially harder and currently underfit with the short run budget.
- Metrics are now realistic for this setup and provide a valid baseline for iterative improvement.

## Artifacts
- `task 3/artifacts_large/checkpoints/*`
- `task 3/artifacts_large/logs/hyperparam_results.json`
- `task 3/artifacts_large/logs/task3_metrics.json`
- `task 3/artifacts_large/logs/rollout_episode_metrics.csv`
- `task 3/artifacts_large/plots/*`
- `task 3/artifacts_large/videos/*`

## Notes
- The `bs=16` sweep path was unstable on this machine during this run window; this report uses the stable `bs=8` sweep.
- The results here supersede prior report versions that were based on synthetic action-coded imagery.
