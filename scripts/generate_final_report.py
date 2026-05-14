#!/usr/bin/env python3
"""Generate comprehensive final results report with visualizations.

Collects all task artifacts and produces:
- results/final_report.md  (comprehensive markdown report)
- results/summary_dashboard.png  (multi-panel summary figure)
- Copies key artifacts into results/ subdirectories
"""

import json
import os
import shutil
import glob
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def safe_copy(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            return True
        except shutil.SameFileError:
            return True
    return False


def copy_artifacts():
    """Copy key artifacts from task directories into results/."""
    base = os.path.dirname(RESULTS_DIR)

    copies = [
        # Task 3
        ("task 3/artifacts_large/logs/task3_metrics.json", "results/task3/task3_metrics.json"),
        ("task 3/artifacts_large/logs/hyperparam_results.json", "results/task3/hyperparam_results.json"),
        ("task 3/artifacts_large/logs/hyperparam_results.csv", "results/task3/hyperparam_results.csv"),
        ("task 3/artifacts_large/logs/rollout_episode_metrics.csv", "results/task3/rollout_episode_metrics.csv"),
        ("task 3/artifacts_large/plots/hyperparam_comparison.png", "results/task3/hyperparam_comparison.png"),
        ("task 3/artifacts_large/images/sr_spl_summary.png", "results/task3/sr_spl_summary.png"),
        # Task 4
        ("task 4/artifacts/ablation_results.json", "results/task4/ablation_results.json"),
        ("task 4/artifacts/ablation_comparison.png", "results/task4/ablation_comparison.png"),
        ("task 4/artifacts/generalization_results.json", "results/task4/generalization_results.json"),
        ("task 4/artifacts/generalization_analysis.png", "results/task4/generalization_analysis.png"),
        # Task 5
        ("task 5/artifacts/task5_results.json", "results/task5/task5_results.json"),
        ("task 5/artifacts/task5_comparison.png", "results/task5/task5_comparison.png"),
        ("task 5/artifacts/task5_summary.md", "results/task5/task5_summary.md"),
    ]

    # Copy confusion matrices from task3
    for f in glob.glob(os.path.join(base, "task 3/artifacts_large/plots/confusion_*.png")):
        name = os.path.basename(f)
        copies.append((f"task 3/artifacts_large/plots/{name}", f"results/task3/{name}"))

    # Copy videos from task3
    for f in glob.glob(os.path.join(base, "task 3/artifacts_large/videos/*.gif")):
        name = os.path.basename(f)
        copies.append((f"task 3/artifacts_large/videos/{name}", f"results/task3/videos/{name}"))

    # Copy learning curves from each task3 run
    for d in glob.glob(os.path.join(base, "task 3/artifacts_large/checkpoints/*/learning_curves.png")):
        run = os.path.basename(os.path.dirname(d))
        copies.append((f"task 3/artifacts_large/checkpoints/{run}/learning_curves.png",
                        f"results/task3/learning_curves_{run}.png"))

    copied = 0
    for src_rel, dst_rel in copies:
        src = os.path.join(base, src_rel)
        dst = os.path.join(base, dst_rel)
        if safe_copy(src, dst):
            copied += 1
    print(f"  Copied {copied} artifacts to results/")


def generate_dashboard():
    """Generate multi-panel summary dashboard."""
    base = os.path.dirname(RESULTS_DIR)

    # Load all metrics
    t2 = load_json(os.path.join(RESULTS_DIR, "task2", "metrics.json"))
    t3 = load_json(os.path.join(RESULTS_DIR, "task3", "task3_metrics.json"))
    t3_hp = load_json(os.path.join(RESULTS_DIR, "task3", "hyperparam_results.json"))
    t4_abl = load_json(os.path.join(RESULTS_DIR, "task4", "ablation_results.json"))
    t5 = load_json(os.path.join(RESULTS_DIR, "task5", "task5_results.json"))

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Vision-Language Navigation — Final Results Dashboard",
                 fontsize=20, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35,
                           top=0.93, bottom=0.05, left=0.05, right=0.97)

    colors = ["#2a9d8f", "#e76f51", "#264653", "#f4a261", "#e9c46a", "#457b9d"]

    # Panel 1: Task 2 Learning Curves
    ax = fig.add_subplot(gs[0, 0:2])
    if t2 and "history" in t2:
        h = t2["history"]
        ep = h.get("epoch", list(range(1, len(h["train_loss"])+1)))
        ax.plot(ep, h["train_loss"], "o-", lw=2, ms=4, color=colors[0], label="Train Loss")
        ax.plot(ep, h["val_loss"], "s-", lw=2, ms=4, color=colors[1], label="Val Loss")
        ax2 = ax.twinx()
        ax2.plot(ep, h["val_acc"], "^-", lw=2, ms=5, color=colors[3], label="Val Acc", alpha=0.8)
        ax2.set_ylabel("Accuracy", color=colors[3], fontweight="bold")
        ax2.set_ylim([0, 1])
        ax2.legend(loc="center right", fontsize=8)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title("Task 2: Baseline Training Curves", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # Panel 2: Task 3 Hyperparameter Comparison
    ax = fig.add_subplot(gs[0, 2:4])
    if t3_hp:
        names = [r["run_name"].replace("_", "\n") for r in t3_hp]
        accs = [r["best_val_acc"] for r in t3_hp]
        c = [colors[i % len(colors)] for i in range(len(names))]
        bars = ax.bar(range(len(names)), accs, color=c, edgecolor="black", lw=1, alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Best Val Accuracy", fontweight="bold")
    ax.set_title("Task 3: Hyperparameter Sweep", fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.2)

    # Panel 3: Task 3 SR/SPL
    ax = fig.add_subplot(gs[1, 0])
    if t3:
        metrics_names = ["SR", "SPL", "OSR"]
        metrics_vals = [t3.get("SR", 0), t3.get("SPL", 0), t3.get("OSR", 0)]
        bars = ax.bar(metrics_names, metrics_vals, color=[colors[0], colors[1], colors[3]],
                      edgecolor="black", lw=1.5, alpha=0.85)
        for bar, val in zip(bars, metrics_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.set_title("Task 3: Navigation Metrics", fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.2)

    # Panel 4: Task 4 Ablation
    ax = fig.add_subplot(gs[1, 1:3])
    if t4_abl:
        configs = list(t4_abl.keys())
        abl_accs = [t4_abl[c].get("best_val_acc", 0) for c in configs]
        c = [colors[i % len(colors)] for i in range(len(configs))]
        bars = ax.bar(configs, abl_accs, color=c, edgecolor="black", lw=1, alpha=0.85)
        for bar, acc in zip(bars, abl_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Best Val Accuracy", fontweight="bold")
    ax.set_title("Task 4: Ablation Study", fontweight="bold", fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.2)

    # Panel 5: Task 4 Generalization
    ax = fig.add_subplot(gs[1, 3])
    t4_gen = load_json(os.path.join(RESULTS_DIR, "task4", "generalization_results.json"))
    if t4_gen:
        scenarios = []
        gen_accs = []
        for k in ["baseline", "unseen_env", "paraphrased"]:
            if k in t4_gen:
                scenarios.append(k.replace("_", "\n"))
                gen_accs.append(t4_gen[k].get("accuracy", 0))
        if scenarios:
            bars = ax.barh(scenarios, gen_accs, color=[colors[0], colors[1], colors[3]][:len(scenarios)],
                          edgecolor="black", lw=1, alpha=0.85)
            for bar, acc in zip(bars, gen_accs):
                ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2,
                        f"{acc:.3f}", ha="right", va="center", fontweight="bold", fontsize=10)
    ax.set_xlim([0, 1.1])
    ax.set_title("Task 4: Generalization", fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.2)

    # Panel 6: Task 5 Comparison
    ax = fig.add_subplot(gs[2, 0:2])
    if t5 and isinstance(t5, list) and len(t5) >= 2:
        names_5 = [r["name"] for r in t5]
        accs_5 = [r["best_val_acc"] for r in t5]
        params_5 = [r["trainable_params"] / 1e6 for r in t5]
        x = np.arange(len(names_5))
        w = 0.35
        b1 = ax.bar(x - w/2, accs_5, w, label="Val Accuracy", color=colors[0], edgecolor="black")
        ax2 = ax.twinx()
        b2 = ax2.bar(x + w/2, params_5, w, label="Params (M)", color=colors[3], edgecolor="black", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names_5)
        ax.set_ylabel("Accuracy", fontweight="bold")
        ax2.set_ylabel("Trainable Params (M)", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Task 5: Baseline vs Lightweight", fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.2)

    # Panel 7: Summary Text
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis("off")
    summary_lines = ["FINAL RESULTS SUMMARY", "=" * 40, ""]
    if t2:
        summary_lines.append(f"Task 2 Best Val Accuracy:  {t2.get('best_val_acc', 0):.4f}")
    if t3:
        summary_lines.append(f"Task 3 Best Val Accuracy:  {t3.get('best_run', {}).get('best_val_acc', 0):.4f}")
        summary_lines.append(f"Task 3 Success Rate (SR):  {t3.get('SR', 0):.1%}")
        summary_lines.append(f"Task 3 SPL:                {t3.get('SPL', 0):.1%}")
        summary_lines.append(f"Task 3 Oracle SR:          {t3.get('OSR', 0):.1%}")
        summary_lines.append(f"Task 3 Nav Error:          {t3.get('NE', 0):.2f}m")
    if t4_abl:
        best_abl = max(t4_abl.values(), key=lambda x: x.get("best_val_acc", 0))
        summary_lines.append(f"Task 4 Best Ablation Acc:  {best_abl.get('best_val_acc', 0):.4f}")
    if t5 and isinstance(t5, list):
        for r in t5:
            summary_lines.append(f"Task 5 {r['name']:12s} Acc: {r['best_val_acc']:.4f}")
    summary_lines.extend(["", "=" * 40,
                          f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"])

    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", alpha=0.9))

    out = os.path.join(RESULTS_DIR, "summary_dashboard.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Dashboard saved: {out}")


def generate_markdown_report():
    """Generate comprehensive final_report.md."""
    t2 = load_json(os.path.join(RESULTS_DIR, "task2", "metrics.json"))
    t3 = load_json(os.path.join(RESULTS_DIR, "task3", "task3_metrics.json"))
    t3_hp = load_json(os.path.join(RESULTS_DIR, "task3", "hyperparam_results.json"))
    t4_abl = load_json(os.path.join(RESULTS_DIR, "task4", "ablation_results.json"))
    t4_gen = load_json(os.path.join(RESULTS_DIR, "task4", "generalization_results.json"))
    t5 = load_json(os.path.join(RESULTS_DIR, "task5", "task5_results.json"))

    lines = []
    lines.append("# Vision-Language Navigation: Final Results Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Author:** Balveer Singh (balveer.25aiz0011@iitrpr.ac.in)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary Dashboard")
    lines.append("![Summary Dashboard](summary_dashboard.png)")
    lines.append("")

    # Task 2
    lines.append("## Task 2: Vision-Language Model Implementation")
    lines.append("")
    if t2:
        lines.append(f"- **Best Validation Accuracy:** {t2.get('best_val_acc', 0):.4f}")
        lines.append(f"- **Final Validation Loss:** {t2.get('final_val_loss', 0):.4f}")
        lines.append(f"- **Training Time:** {t2.get('training_time_sec', 0)/60:.1f} minutes")
        lines.append(f"- **Total Parameters:** {t2.get('total_params', 0):,}")
        lines.append(f"- **Trainable Parameters:** {t2.get('trainable_params', 0):,}")
        lines.append(f"- **GPU:** {t2.get('gpu_name', 'N/A')}")
        cfg = t2.get("config", {})
        lines.append(f"- **Config:** epochs={cfg.get('epochs')}, lr={cfg.get('lr')}, "
                      f"batch_size={cfg.get('batch_size')}, label_smoothing={cfg.get('label_smoothing')}")
        lines.append("")
        pc = t2.get("per_class_accuracy", {})
        if pc:
            lines.append("### Per-Class Accuracy")
            lines.append("| Action | Accuracy |")
            lines.append("|--------|----------|")
            for action, acc in pc.items():
                lines.append(f"| {action} | {acc:.4f} |")
            lines.append("")
        lines.append("### Learning Curves")
        lines.append("![Task 2 Learning Curves](task2/learning_curves.png)")
        lines.append("")
        lines.append("### Confusion Matrix")
        lines.append("![Task 2 Confusion Matrix](task2/confusion_matrix.png)")
    else:
        lines.append("*Results not available*")
    lines.append("")

    # Task 3
    lines.append("---")
    lines.append("## Task 3: Baseline Training and Evaluation")
    lines.append("")
    if t3:
        best = t3.get("best_run", {})
        lines.append(f"- **Best Run:** {best.get('run_name', 'N/A')}")
        lines.append(f"- **Best Validation Accuracy:** {best.get('best_val_acc', 0):.4f}")
        lines.append(f"- **Success Rate (SR):** {t3.get('SR', 0):.1%}")
        lines.append(f"- **SPL:** {t3.get('SPL', 0):.1%}")
        lines.append(f"- **Oracle Success Rate:** {t3.get('OSR', 0):.1%}")
        lines.append(f"- **Navigation Error:** {t3.get('NE', 0):.2f}m")
        lines.append(f"- **Eval Episodes:** {t3.get('num_eval_episodes', 0)}")
        lines.append("")

    if t3_hp:
        lines.append("### Hyperparameter Sweep Results")
        lines.append("| Run | LR | Batch Size | Best Val Acc | Final Val Acc | Final Val Loss |")
        lines.append("|-----|-----|------------|-------------|---------------|----------------|")
        for r in t3_hp:
            lines.append(f"| {r['run_name']} | {r.get('lr', 'N/A')} | "
                          f"{r.get('batch_size', 'N/A')} | {r['best_val_acc']:.4f} | "
                          f"{r['final_val_acc']:.4f} | {r['final_val_loss']:.4f} |")
        lines.append("")

    lines.append("### Hyperparameter Comparison")
    lines.append("![Hyperparameter Comparison](task3/hyperparam_comparison.png)")
    lines.append("")
    lines.append("### SR/SPL Summary")
    lines.append("![SR SPL](task3/sr_spl_summary.png)")
    lines.append("")

    # Confusion matrices
    cms = sorted(glob.glob(os.path.join(RESULTS_DIR, "task3", "confusion_*.png")))
    if cms:
        lines.append("### Confusion Matrices")
        for cm_path in cms:
            name = os.path.basename(cm_path).replace("confusion_", "").replace(".png", "")
            lines.append(f"#### {name}")
            lines.append(f"![Confusion {name}](task3/{os.path.basename(cm_path)})")
            lines.append("")

    # Videos
    vids = sorted(glob.glob(os.path.join(RESULTS_DIR, "task3", "videos", "*.gif")))
    if vids:
        lines.append("### Navigation Rollout Videos")
        lines.append(f"Generated {len(vids)} rollout GIFs in `results/task3/videos/`")
        lines.append("")

    # Task 4
    lines.append("---")
    lines.append("## Task 4: Generalization and Ablation Study")
    lines.append("")

    if t4_abl:
        lines.append("### Ablation Study Results")
        lines.append("| Configuration | Trainable Params | Best Val Acc | Final Val Acc |")
        lines.append("|---------------|-----------------|-------------|---------------|")
        for config_name, data in t4_abl.items():
            lines.append(f"| {data.get('config', config_name)} | "
                          f"{data.get('trainable_params', 0):,} | "
                          f"{data.get('best_val_acc', 0):.4f} | "
                          f"{data.get('final_val_acc', 0):.4f} |")
        lines.append("")
        lines.append("![Ablation Comparison](task4/ablation_comparison.png)")
        lines.append("")

    if t4_gen:
        lines.append("### Generalization Evaluation")
        lines.append("| Scenario | Accuracy | Loss |")
        lines.append("|----------|----------|------|")
        for scenario in ["baseline", "unseen_env", "paraphrased"]:
            if scenario in t4_gen:
                d = t4_gen[scenario]
                lines.append(f"| {d.get('description', scenario)} | "
                              f"{d.get('accuracy', 0):.4f} | {d.get('loss', 0):.4f} |")
        lines.append("")
        if "data_scaling" in t4_gen and t4_gen["data_scaling"]:
            lines.append("### Data Scaling Analysis")
            lines.append("| Training Data % | Accuracy | Loss |")
            lines.append("|----------------|----------|------|")
            for pct, vals in sorted(t4_gen["data_scaling"].items(), key=lambda x: int(x[0])):
                lines.append(f"| {pct}% | {vals.get('accuracy', 0):.4f} | {vals.get('loss', 0):.4f} |")
            lines.append("")
        lines.append("![Generalization Analysis](task4/generalization_analysis.png)")
        lines.append("")

    # Task 5
    lines.append("---")
    lines.append("## Task 5: Controlled Extension")
    lines.append("")
    if t5 and isinstance(t5, list) and len(t5) >= 2:
        baseline = t5[0]
        lightweight = t5[1]
        param_reduction = 100.0 * (1.0 - lightweight["trainable_params"] / max(baseline["trainable_params"], 1))
        acc_delta = lightweight["best_val_acc"] - baseline["best_val_acc"]
        speedup = baseline.get("mean_epoch_time_sec", 1) / max(lightweight.get("mean_epoch_time_sec", 1), 1e-9)

        lines.append("### Baseline vs Lightweight Variant")
        lines.append("| Metric | Baseline | Lightweight | Change |")
        lines.append("|--------|----------|-------------|--------|")
        lines.append(f"| Fusion Dim | {baseline['fusion_dim']} | {lightweight['fusion_dim']} | |")
        lines.append(f"| Num Heads | {baseline['num_heads']} | {lightweight['num_heads']} | |")
        lines.append(f"| Dropout | {baseline['dropout']} | {lightweight['dropout']} | |")
        lines.append(f"| Trainable Params | {baseline['trainable_params']:,} | "
                      f"{lightweight['trainable_params']:,} | {param_reduction:+.1f}% |")
        lines.append(f"| Best Val Acc | {baseline['best_val_acc']:.4f} | "
                      f"{lightweight['best_val_acc']:.4f} | {acc_delta:+.4f} |")
        lines.append(f"| Mean Epoch Time | {baseline.get('mean_epoch_time_sec', 0):.1f}s | "
                      f"{lightweight.get('mean_epoch_time_sec', 0):.1f}s | {speedup:.2f}x |")
        lines.append("")
        lines.append("![Task 5 Comparison](task5/task5_comparison.png)")
    else:
        lines.append("*Results not available*")
    lines.append("")

    # Conclusion
    lines.append("---")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("This report presents a comprehensive evaluation of vision-language navigation ")
    lines.append("using a CLIP-based cross-attention policy trained with imitation learning. ")
    lines.append("Key findings:")
    lines.append("")
    if t3:
        lines.append(f"1. **Navigation Performance:** Achieved SR={t3.get('SR', 0):.1%}, "
                      f"SPL={t3.get('SPL', 0):.1%} on evaluation episodes")
    if t3_hp:
        best_run = t3_hp[0]
        lines.append(f"2. **Best Configuration:** {best_run['run_name']} with "
                      f"accuracy={best_run['best_val_acc']:.4f}")
    if t4_abl:
        configs = list(t4_abl.keys())
        best_abl = max(configs, key=lambda c: t4_abl[c].get("best_val_acc", 0))
        lines.append(f"3. **Ablation Finding:** {best_abl} achieved best accuracy "
                      f"({t4_abl[best_abl].get('best_val_acc', 0):.4f})")
    if t5 and isinstance(t5, list) and len(t5) >= 2:
        lines.append(f"4. **Extension Result:** Lightweight variant achieves "
                      f"{param_reduction:.1f}% parameter reduction with "
                      f"{'comparable' if abs(acc_delta) < 0.05 else 'different'} accuracy")
    lines.append("")

    report_path = os.path.join(RESULTS_DIR, "final_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Report saved: {report_path}")


def main():
    print("=" * 60)
    print("Generating Final Results Report")
    print("=" * 60)

    for d in ["task2", "task3", "task3/videos", "task4", "task5"]:
        os.makedirs(os.path.join(RESULTS_DIR, d), exist_ok=True)

    print("\n1. Copying artifacts...")
    copy_artifacts()

    print("\n2. Generating dashboard...")
    generate_dashboard()

    print("\n3. Generating markdown report...")
    generate_markdown_report()

    print(f"\n{'=' * 60}")
    print(f"✅ Final report ready at: {RESULTS_DIR}/final_report.md")
    print(f"   Dashboard at: {RESULTS_DIR}/summary_dashboard.png")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
