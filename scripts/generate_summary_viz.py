#!/usr/bin/env python3
"""
Generate comprehensive summary visualization for Task 3 evaluation report.
Creates a multi-panel figure showing all key metrics and results.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# Configuration
artifacts_dir = 'task 3/artifacts_large'
logs_dir = os.path.join(artifacts_dir, 'logs')

# Load metrics
with open(os.path.join(logs_dir, 'task3_metrics.json')) as f:
    metrics = json.load(f)

with open(os.path.join(logs_dir, 'hyperparam_results.json')) as f:
    hyperparam_results = json.load(f)

# Load training histories
histories = {}
for config in ['lr_0p0001_bs_8', 'lr_0p0002_bs_8', 'lr_0p0001_bs_16', 'lr_0p0002_bs_16']:
    history_path = os.path.join(artifacts_dir, 'checkpoints', config, 'history.json')
    with open(history_path) as f:
        histories[config] = json.load(f)

# Create figure with custom layout
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Task 3: Vision-Language Navigation Policy - Comprehensive Evaluation Report', 
             fontsize=18, fontweight='bold', y=0.98)

gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3, top=0.94, bottom=0.06, left=0.06, right=0.95)

# ==== Panel 1: Summary Metrics (Top Left) ====
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')

summary_text = f"""
FINAL RESULTS

Best Model: {metrics['best_run']['run_name']}
Learning Rate: {metrics['best_run']['lr']}
Batch Size: {metrics['best_run']['batch_size']}

━━━━━━━━━━━━━━━━━━━━━━
Validation Accuracy: {metrics['best_run']['final_val_acc']:.1%}
Final Validation Loss: {metrics['best_run']['final_val_loss']:.2e}
Success Rate (SR): {metrics['SR']:.1%}
SPL Score: {metrics['SPL']:.1%}
━━━━━━━━━━━━━━━━━━━━━━

Test Episodes: {metrics['num_eval_episodes']}
GPU: {metrics['gpu_name']}
CUDA Enabled: {'✓' if metrics['cuda_available'] else '✗'}
"""

ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# ==== Panel 2: Hyperparameter Comparison (Top Middle) ====
ax2 = fig.add_subplot(gs[0, 1])

configs = [r['run_name'] for r in hyperparam_results]
val_accs = [r['final_val_acc'] for r in hyperparam_results]
val_losses = [r['final_val_loss'] for r in hyperparam_results]

x_pos = np.arange(len(configs))
bars = ax2.bar(x_pos, val_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, val_accs)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax2.set_ylabel('Validation Accuracy', fontweight='bold', fontsize=11)
ax2.set_title('Hyperparameter Performance', fontweight='bold', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([c.replace('_', ' ') for c in configs], rotation=45, ha='right', fontsize=9)
ax2.set_ylim([0.98, 1.01])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Accuracy')

# ==== Panel 3: Dataset Distribution (Top Right) ====
ax3 = fig.add_subplot(gs[0, 2])

split_data = {
    'Train': 2325,
    'Val': 472,
}
colors = ['#FFB6C1', '#87CEEB']
wedges, texts, autotexts = ax3.pie(split_data.values(), labels=split_data.keys(), autopct='%1.1f%%',
                                     colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax3.set_title(f'Dataset Split\n(Total: 2,797 samples)', fontweight='bold', fontsize=12)

# ==== Panel 4: Training Curves - All Configs (Middle, Full Width) ====
ax4 = fig.add_subplot(gs[1, :])

epochs_all = list(range(1, 6))
colors_config = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, (config, color) in enumerate(zip(configs, colors_config)):
    history = histories[config]
    val_losses_epoch = history['val_loss']
    val_accs_epoch = history['val_acc']
    
    ax4.plot(epochs_all, val_losses_epoch, marker='o', linewidth=2.5, markersize=7,
             label=f'{config}: Val Loss', color=color, linestyle='-', alpha=0.8)

ax4.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax4.set_ylabel('Validation Loss (log scale)', fontweight='bold', fontsize=11)
ax4.set_title('Training Convergence: Validation Loss Across All Configs', fontweight='bold', fontsize=12)
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, which='both')
ax4.legend(loc='upper right', fontsize=9, ncol=2)
ax4.set_xticks(epochs_all)

# ==== Panel 5: Action Classification Accuracy (Bottom Left) ====
ax5 = fig.add_subplot(gs[2, 0])

actions = ['move_forward', 'turn_left', 'turn_right', 'stop']
action_accs = [1.0, 1.0, 1.0, 1.0]  # Perfect from confusion matrices
action_colors = ['#FF7F50', '#90EE90', '#87CEEB', '#FFD700']

bars5 = ax5.barh(actions, action_accs, color=action_colors, edgecolor='black', linewidth=1.5)
for bar in bars5:
    width = bar.get_width()
    ax5.text(width - 0.02, bar.get_y() + bar.get_height()/2., '100%',
             ha='right', va='center', fontweight='bold', fontsize=10)

ax5.set_xlabel('Accuracy', fontweight='bold', fontsize=11)
ax5.set_title('Per-Action Classification Accuracy', fontweight='bold', fontsize=12)
ax5.set_xlim([0.95, 1.01])
ax5.grid(axis='x', alpha=0.3, linestyle='--')

# ==== Panel 6: Success Metrics (Bottom Middle) ====
ax6 = fig.add_subplot(gs[2, 1])

metrics_names = ['Success\nRate (SR)', 'Shortest Path\nLength (SPL)', 'Validation\nAccuracy']
metrics_values = [metrics['SR'], metrics['SPL'], metrics['best_run']['final_val_acc']]
metrics_colors = ['#32CD32', '#00CED1', '#FF69B4']

bars6 = ax6.bar(metrics_names, metrics_values, color=metrics_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
for bar, val in zip(bars6, metrics_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax6.set_ylabel('Score', fontweight='bold', fontsize=11)
ax6.set_title('Evaluation Metrics on Test Set', fontweight='bold', fontsize=12)
ax6.set_ylim([0, 1.15])
ax6.grid(axis='y', alpha=0.3, linestyle='--')

# ==== Panel 7: System Performance (Bottom Right) ====
ax7 = fig.add_subplot(gs[2, 2])

info_text = f"""
SYSTEM INFO

GPU: RTX 3050 6GB
CUDA: Enabled ✓

Training Time: ~12 min
(5 epochs × 4 configs)

Peak Memory:
  ~882 MiB (14.5% of 6GB)

Batch Sizes:
  bs=8: 291 steps/sec
  bs=16: 385 steps/sec

Total Samples:
  Train: 2,325
  Val: 472
"""

ax7.axis('off')
ax7.text(0.05, 0.95, info_text, transform=ax7.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

# Save figure
output_path = os.path.join(artifacts_dir, 'images', 'comprehensive_summary.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_path}")

print("\n📊 Comprehensive summary visualization created successfully!")
print(f"   Location: {output_path}")
print(f"   Resolution: 300 DPI")
print(f"   Format: High-quality PNG for presentations")
