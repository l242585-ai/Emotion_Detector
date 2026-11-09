"""
Visualize Training Results — Member 2
Generates all required plots for the report:
  1. Training curves (loss & accuracy) for BiLSTM and DistilBERT
  2. Model comparison bar chart (accuracy + F1 macro)
  3. Hyperparameter sensitivity plots
  4. Per-class F1 score comparison

Usage:
    python src/visualize_results.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EMOTION_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
EMOTION_COLORS = ['#5B8DEF', '#F4C842', '#FF6B9D', '#FF5733', '#8E44AD', '#2ECC71']

# ── Consistent styling ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': '#0F1117',
    'axes.facecolor': '#1A1D2E',
    'axes.labelcolor': '#E0E0E0',
    'axes.edgecolor': '#2A2D3E',
    'text.color': '#E0E0E0',
    'xtick.color': '#B0B0B0',
    'ytick.color': '#B0B0B0',
    'grid.color': '#2A2D3E',
    'grid.alpha': 0.6,
    'lines.linewidth': 2.0,
    'legend.facecolor': '#1A1D2E',
    'legend.edgecolor': '#2A2D3E',
    'legend.labelcolor': '#E0E0E0',
})

MODEL_COLORS = {
    'BiLSTM':    {'loss': '#5B8DEF', 'acc': '#7EC8F4'},
    'DistilBERT': {'loss': '#FF6B9D', 'acc': '#FFB347'},
}


# ─────────────────────────────────────────────
# 1. Training Curves
# ─────────────────────────────────────────────

def plot_training_curves(out_path='results/training_curves.png'):
    """
    Plot training and validation loss/accuracy curves for all deep models.
    Loads from results/{model}_history.json files.
    """
    models_to_plot = []

    for model_name, history_file in [
        ('BiLSTM', 'results/bilstm_history.json'),
        ('DistilBERT', 'results/distilbert_history.json')
    ]:
        if os.path.exists(history_file):
            with open(history_file) as f:
                history = json.load(f)
            models_to_plot.append((model_name, history))

    if not models_to_plot:
        print("  ⚠️  No training history files found. Train models first.")
        return

    n_models = len(models_to_plot)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Training & Validation Curves', fontsize=18, fontweight='bold',
                 color='white', y=1.01)

    for row_idx, (model_name, history) in enumerate(models_to_plot):
        ax_loss, ax_acc = axes[row_idx]
        colors = MODEL_COLORS.get(model_name, {'loss': '#5B8DEF', 'acc': '#FF6B9D'})
        epochs = range(1, len(history['train_loss']) + 1)

        # Loss subplot
        ax_loss.plot(epochs, history['train_loss'], color=colors['loss'],
                     label='Train Loss', linewidth=2.5, marker='o', markersize=4)
        ax_loss.plot(epochs, history['val_loss'], color=colors['loss'],
                     label='Val Loss', linewidth=2.5, linestyle='--',
                     marker='s', markersize=4, alpha=0.8)
        ax_loss.fill_between(epochs, history['train_loss'], history['val_loss'],
                             alpha=0.1, color=colors['loss'])
        ax_loss.set_title(f'{model_name} — Loss', fontsize=13, fontweight='bold',
                          color='white', pad=10)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(loc='upper right')
        ax_loss.grid(True, alpha=0.4)

        # Mark best epoch (lowest val loss)
        best_epoch = int(np.argmin(history['val_loss'])) + 1
        best_val_loss = min(history['val_loss'])
        ax_loss.axvline(x=best_epoch, color='#F4C842', linestyle=':', linewidth=1.5,
                        label=f'Best epoch ({best_epoch})')
        ax_loss.scatter([best_epoch], [best_val_loss], color='#F4C842',
                        zorder=5, s=80, marker='*')
        ax_loss.legend(loc='upper right')

        # Accuracy subplot
        ax_acc.plot(epochs, history['train_acc'], color=colors['acc'],
                    label='Train Acc', linewidth=2.5, marker='o', markersize=4)
        ax_acc.plot(epochs, history['val_acc'], color=colors['acc'],
                    label='Val Acc', linewidth=2.5, linestyle='--',
                    marker='s', markersize=4, alpha=0.8)
        ax_acc.fill_between(epochs, history['train_acc'], history['val_acc'],
                            alpha=0.1, color=colors['acc'])
        ax_acc.set_title(f'{model_name} — Accuracy', fontsize=13, fontweight='bold',
                         color='white', pad=10)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_ylim(0, 1.05)
        ax_acc.grid(True, alpha=0.4)

        best_acc = max(history['val_acc'])
        ax_acc.axhline(y=best_acc, color='#F4C842', linestyle=':', linewidth=1.5,
                       label=f'Best val acc ({best_acc:.3f})')
        ax_acc.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    plt.close()
    print(f"✅ Training curves → {out_path}")


# ─────────────────────────────────────────────
# 2. Model Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_model_comparison(out_path='results/model_comparison.png'):
    """
    Bar chart comparing all 3 models on accuracy and F1 macro.
    Reads from results/metrics_summary.json.
    """
    if not os.path.exists('results/metrics_summary.json'):
        print("  ⚠️  metrics_summary.json not found. Run evaluate_all.py first.")
        return

    with open('results/metrics_summary.json') as f:
        summary = json.load(f)

    models_data = summary['models']
    model_names = list(models_data.keys())
    accuracies = [models_data[m]['accuracy'] for m in model_names]
    f1_macros = [models_data[m]['f1_macro'] for m in model_names]
    f1_weighted = [models_data[m]['f1_weighted'] for m in model_names]

    bar_colors = ['#5B8DEF', '#FF6B9D', '#F4C842', '#2ECC71']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold',
                 color='white', y=1.02)

    metric_data = [
        ('Test Accuracy', accuracies),
        ('F1 Score (Macro)', f1_macros),
        ('F1 Score (Weighted)', f1_weighted)
    ]

    for ax, (title, values) in zip(axes, metric_data):
        bars = ax.bar(model_names, values, color=bar_colors[:len(model_names)],
                      width=0.5, edgecolor='#2A2D3E', linewidth=1.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='white'
            )

        ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.4)
        ax.set_axisbelow(True)

        # Highlight best model
        best_idx = int(np.argmax(values))
        bars[best_idx].set_edgecolor('#F4C842')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    plt.close()
    print(f"✅ Model comparison chart → {out_path}")


# ─────────────────────────────────────────────
# 3. Hyperparameter Sensitivity
# ─────────────────────────────────────────────

def plot_hyperparameter_sensitivity(out_path='results/hyperparameter_sensitivity.png'):
    """
    Plot how val_acc changes with each hyperparameter for BiLSTM.
    Reads from experiments/hyperparameter_log.csv.
    """
    log_path = 'experiments/hyperparameter_log.csv'
    if not os.path.exists(log_path):
        print("  ⚠️  hyperparameter_log.csv not found. Run tuning first.")
        return

    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16,
                 fontweight='bold', color='white', y=1.02)

    for model_name, model_df in df.groupby('model'):
        color = '#5B8DEF' if model_name == 'BiLSTM' else '#FF6B9D'

        # LR sensitivity
        lr_df = model_df[model_df['experiment'].str.contains('lr_')].sort_values('lr')
        if not lr_df.empty:
            axes[0].plot(
                [str(x) for x in lr_df['lr']], lr_df['val_acc'],
                marker='o', label=model_name, color=color, linewidth=2.5
            )

        # Batch sensitivity
        batch_df = model_df[model_df['experiment'].str.contains('batch_')].sort_values('batch_size')
        if not batch_df.empty:
            axes[1].plot(
                [str(x) for x in batch_df['batch_size']], batch_df['val_acc'],
                marker='s', label=model_name, color=color, linewidth=2.5
            )

        # Dropout sensitivity
        drop_df = model_df[model_df['experiment'].str.contains('dropout_')].sort_values('dropout')
        if not drop_df.empty:
            axes[2].plot(
                [str(x) for x in drop_df['dropout']], drop_df['val_acc'],
                marker='^', label=model_name, color=color, linewidth=2.5
            )

    titles = ['Learning Rate', 'Batch Size', 'Dropout Rate']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=10)
        ax.set_ylabel('Validation Accuracy')
        ax.set_xlabel(title)
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    plt.close()
    print(f"✅ Hyperparameter sensitivity → {out_path}")


# ─────────────────────────────────────────────
# 4. Per-class F1 Comparison
# ─────────────────────────────────────────────

def plot_per_class_f1(out_path='results/per_class_f1.png'):
    """
    Grouped bar chart: per-class F1 score for each model.
    Reads predictions from results/predictions.csv.
    """
    if not os.path.exists('results/predictions.csv'):
        print("  ⚠️  predictions.csv not found. Run evaluate_all.py first.")
        return

    from sklearn.metrics import f1_score

    df = pd.read_csv('results/predictions.csv')
    true_labels = df['true_label'].values

    model_cols = [c for c in df.columns if c.startswith('pred_') and 'emotion' not in c]
    model_names = [c.replace('pred_', '').replace('_', ' ').title() for c in model_cols]

    # Compute per-class F1
    all_f1s = []
    for col in model_cols:
        preds = df[col].values
        f1s = f1_score(true_labels, preds, average=None, labels=list(range(6)))
        all_f1s.append(f1s)

    x = np.arange(len(EMOTION_NAMES))
    width = 0.25
    n_models = len(model_cols)
    offsets = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2, n_models)

    bar_colors = ['#5B8DEF', '#FF6B9D', '#F4C842']

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#1A1D2E')

    for i, (name, f1s, offset) in enumerate(zip(model_names, all_f1s, offsets)):
        bars = ax.bar(x + offset, f1s, width, label=name,
                      color=bar_colors[i % len(bar_colors)],
                      edgecolor='#0F1117', linewidth=0.8, alpha=0.9)

    ax.set_xlabel('Emotion Class', fontsize=13)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('Per-Class F1 Score — All Models', fontsize=16,
                 fontweight='bold', color='white', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in EMOTION_NAMES], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    plt.close()
    print(f"✅ Per-class F1 chart → {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    os.makedirs('results', exist_ok=True)

    print("\n[1/4] Training curves...")
    plot_training_curves()

    print("\n[2/4] Model comparison bar chart...")
    plot_model_comparison()

    print("\n[3/4] Hyperparameter sensitivity plots...")
    plot_hyperparameter_sensitivity()

    print("\n[4/4] Per-class F1 scores...")
    plot_per_class_f1()

    print("\n" + "=" * 60)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    for f in [
        'results/training_curves.png',
        'results/model_comparison.png',
        'results/hyperparameter_sensitivity.png',
        'results/per_class_f1.png'
    ]:
        status = "✅" if os.path.exists(f) else "⚠️  (run prereq steps first)"
        print(f"  {status} {f}")


if __name__ == '__main__':
    main()
