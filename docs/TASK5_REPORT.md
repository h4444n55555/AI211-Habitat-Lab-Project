# Task 5: Controlled Extension Report

Date: May 14, 2026  
Data root: `task 3/data_large`

## Summary
Task 5 compares two policy variants under the same training regime:
- Baseline fusion block
- Lightweight fusion block (smaller fusion_dim, fewer heads, higher dropout)

Run command:
- `task 5/scripts/task5_controlled_extension.py --data-root "task 3/data_large" --output-dir "task 5/artifacts" --epochs 1 --batch-size 8 --num-workers 0 --device auto`

## Results
From `task 5/artifacts/task5_results.json`:

Baseline:
- fusion_dim: 512
- num_heads: 8
- dropout: 0.1
- trainable_params: 5,126,660
- best_val_acc: 0.3795620437956204
- final_val_loss: 1.7322279113342582
- mean_epoch_time_sec: 28.240015907000043

Lightweight:
- fusion_dim: 256
- num_heads: 4
- dropout: 0.2
- trainable_params: 1,449,220
- best_val_acc: 0.30170316301703165
- final_val_loss: 1.5531223609209641
- mean_epoch_time_sec: 26.368161697000005

Trainable parameter reduction:
- 71.73% (lightweight vs baseline)

## Interpretation
- The lightweight model is substantially smaller and slightly faster per epoch.
- In this one-epoch run, baseline achieved higher validation accuracy.
- Lightweight model achieved lower final CE loss but worse top-1 action accuracy, suggesting different calibration dynamics.

## Artifacts
- `task 5/artifacts/task5_results.json`
- `task 5/artifacts/task5_summary.md`
- `task 5/artifacts/task5_comparison.png`
- `task 5/artifacts/baseline/*`
- `task 5/artifacts/lightweight/*`
