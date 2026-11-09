# evaluate.py - Comprehensive evaluation of trained emotion classifier
# Member 3: Evaluation & Error Analysis
#
# Usage:
#   python src/evaluate.py
#
# Expects:
#   results/predictions.csv  (from Member 2, columns: true_label, predicted_label, [prob_0..prob_5])
#
# Outputs:
#   results/evaluation_metrics.json
#   results/confusion_matrix.png
#   results/classification_report.txt

import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, '.')
from src.utils.metrics import compute_all_metrics, get_classification_report, EMOTION_LABELS

os.makedirs('results', exist_ok=True)

# Load predictions
print("Loading predictions...")
pred_path = 'results/predictions.csv'

if not os.path.exists(pred_path):
    print("predictions.csv not found - generating dummy predictions for testing")
    np.random.seed(42)
    n = 500
    y_true = np.random.randint(0, 6, n)
    y_pred = np.where(np.random.rand(n) > 0.35, y_true, np.random.randint(0, 6, n))
    probs = np.random.dirichlet(np.ones(6), size=n)
    df = pd.DataFrame({'true_label': y_true, 'predicted_label': y_pred})
    for i in range(6):
        df[f'prob_{i}'] = probs[:, i]
    df.to_csv(pred_path, index=False)
    print("Dummy predictions saved to results/predictions.csv")

df = pd.read_csv(pred_path)
y_true = df['true_label'].values
y_pred = df['predicted_label'].values

prob_cols = [f'prob_{i}' for i in range(6)]
y_prob = df[prob_cols].values if all(c in df.columns for c in prob_cols) else None

print(f"Loaded {len(df)} predictions")

# Compute metrics
print("Computing metrics...")
metrics = compute_all_metrics(y_true, y_pred, y_prob)

# Print summary
print(f"\nPrecision  macro={metrics['precision_macro']:.4f}  weighted={metrics['precision_weighted']:.4f}")
print(f"Recall     macro={metrics['recall_macro']:.4f}  weighted={metrics['recall_weighted']:.4f}")
print(f"F1         macro={metrics['f1_macro']:.4f}  weighted={metrics['f1_weighted']:.4f}")
if metrics['auc_roc_macro']:
    print(f"AUC-ROC    macro={metrics['auc_roc_macro']:.4f}  weighted={metrics['auc_roc_weighted']:.4f}")

print("\nPer-class F1:")
for label in EMOTION_LABELS:
    print(f"  {label:<10}: {metrics[f'f1_{label}']:.4f}")

# Save metrics JSON
metrics_to_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
metrics_to_save['confusion_matrix'] = metrics['confusion_matrix']
with open('results/evaluation_metrics.json', 'w') as f:
    json.dump(metrics_to_save, f, indent=2)
print("\nSaved: results/evaluation_metrics.json")

# Classification report
report = get_classification_report(y_true, y_pred)
with open('results/classification_report.txt', 'w') as f:
    f.write(report)
print("Saved: results/classification_report.txt")
print("\nClassification Report:")
print(report)

# Confusion matrix plot
cm = np.array(metrics['confusion_matrix'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.close()
print("Saved: results/confusion_matrix.png")

# ROC curves (if probabilities available)
if y_prob is not None:
    n_classes = 6
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(EMOTION_LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('results/roc_curves.png', dpi=150)
    plt.close()
    print("Saved: results/roc_curves.png")

print("\nEvaluation complete.")
