# Task 5 Controlled Extension

## Baseline vs Lightweight Variant

- Baseline: fusion_dim=512, num_heads=8, dropout=0.1
- Lightweight: fusion_dim=256, num_heads=4, dropout=0.2
- Trainable parameter reduction: 71.73%
- Validation accuracy change: -0.49 percentage points
- Mean epoch time speedup: 1.06x

## Interpretation

The lightweight variant keeps the frozen-CLIP training regime intact while shrinking the
fusion block and increasing dropout slightly. This tests whether the model is over-parameterized
for the synthetic VLN task and whether a cheaper policy can preserve performance.
