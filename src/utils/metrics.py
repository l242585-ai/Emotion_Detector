# metrics.py - Evaluation metrics for emotion classification
# Member 3: Evaluation & Error Analysis

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize

EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Compute all required metrics and return as a dict."""
    metrics = {}

    # Precision
    metrics['precision_macro']    = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    metrics['precision_micro']    = precision_score(y_true, y_pred, average='micro',    zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    # Recall
    metrics['recall_macro']    = recall_score(y_true, y_pred, average='macro',    zero_division=0)
    metrics['recall_micro']    = recall_score(y_true, y_pred, average='micro',    zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # F1
    metrics['f1_macro']    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    metrics['f1_micro']    = f1_score(y_true, y_pred, average='micro',    zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # AUC-ROC (one-vs-rest, needs probability scores)
    if y_prob is not None:
        n_classes = len(np.unique(y_true))
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        metrics['auc_roc_macro']    = roc_auc_score(y_bin, y_prob, average='macro',    multi_class='ovr')
        metrics['auc_roc_weighted'] = roc_auc_score(y_bin, y_prob, average='weighted', multi_class='ovr')
    else:
        metrics['auc_roc_macro']    = None
        metrics['auc_roc_weighted'] = None

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, label in enumerate(EMOTION_LABELS):
        metrics[f'f1_{label}'] = per_class_f1[i] if i < len(per_class_f1) else 0.0

    return metrics


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=EMOTION_LABELS, zero_division=0)
