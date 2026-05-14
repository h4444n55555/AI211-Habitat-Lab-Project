# Task 4: Generalization Evaluation Report

Date: May 14, 2026  
Base checkpoint: `task 3/artifacts_large/checkpoints/lr_0p0002_bs_8/best.pt`

## Summary
Task 4 was regenerated and evaluated using the new real-scene Task 3 pipeline.
This report replaces prior synthetic-task conclusions.

## Data Generation
Script: `task 4/scripts/task4_generate_data.py`

Generated outputs:
- `task 4/data/unseen_env`
- `task 4/data/paraphrased`
- `task 4/data/reduced_10pct`
- `task 4/data/reduced_25pct`
- `task 4/data/reduced_50pct`

Counts from run:
- Paraphrased: 100 episodes, 2041 samples
- Reduced 10%: 10 episodes, 201 samples
- Reduced 25%: 25 episodes, 525 samples
- Reduced 50%: 50 episodes, 1018 samples

## Evaluation Results
From `task 4/artifacts/generalization_results.json`:

Baseline:
- Accuracy: 0.45985401459854014
- Loss: 1.245924830291683

Unseen environment:
- Accuracy: 0.44768856447688565
- Loss: 1.2470200288034703

Paraphrased instructions:
- Accuracy: 0.32435080842724157
- Loss: 1.3080891582791798

Data scaling subsets (evaluation on subset distributions):
- 10% subset accuracy: 0.44776119402985076
- 25% subset accuracy: 0.44952380952380955
- 50% subset accuracy: 0.481335952848723

## Interpretation
- Unseen-scene performance is close to baseline, indicating moderate scene transfer under this short-run training budget.
- Paraphrase robustness is currently weaker and likely needs more instruction diversity during training.
- The 50% subset split shows slightly better accuracy than 10/25%, consistent with higher data coverage.

## Artifacts
- `task 4/artifacts/generalization_results.json`
- `task 4/artifacts/generalization_analysis.png`

## Notes
- This run uses the current MP3D-backed data generation and updated checkpoint loader.
- These numbers supersede previous report versions that were tied to placeholder synthetic data.
