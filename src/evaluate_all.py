"""
Evaluate All 3 Models — Member 2 Deliverable
Loads each trained model, evaluates on test set, and produces:
  - results/predictions.csv       (all models, combined)
  - results/metrics_summary.json  (accuracy, F1 for all models)
  - results/confusion_matrices.png

Usage:
    python src/evaluate_all.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (works in Colab & VS Code)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_loader import load_emotion_data, calculate_class_weights
from src.utils.preprocessing import TextPreprocessor

EMOTION_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
EMOTION_COLORS = ['#5B8DEF', '#F4C842', '#FF6B9D', '#FF5733', '#8E44AD', '#2ECC71']


# ─────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────

def load_baseline(model_path='models/best_baseline.pkl', data_dir='data/processed'):
    """Load and evaluate the baseline (TF-IDF + LogReg) model."""
    if not os.path.exists(model_path):
        print(f"  ⚠️  Baseline model not found at {model_path}")
        print("     Run: python src/train_baseline.py")
        return None

    from src.models.baseline_model import BaselineModel
    model = BaselineModel.load(model_path)

    _, _, _, _, test_texts, test_labels = load_emotion_data(data_dir)
    preds = model.predict(test_texts)
    probas = model.predict_proba(test_texts)

    return {
        'name': 'TF-IDF + LogReg (Baseline)',
        'short': 'Baseline',
        'test_texts': test_texts,
        'test_true': test_labels,
        'test_preds': preds,
        'test_probas': probas
    }


def load_bilstm(model_path='models/best_bilstm.pth', data_dir='data/processed'):
    """Load and evaluate the BiLSTM model."""
    if not os.path.exists(model_path):
        print(f"  ⚠️  BiLSTM model not found at {model_path}")
        print("     Run: python src/train_deep.py")
        return None

    from src.models.deep_model import BiLSTMEmotionClassifier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = BiLSTMEmotionClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Rebuild preprocessor and load data
    train_texts, train_labels, _, _, test_texts, test_labels = load_emotion_data(data_dir)
    preprocessor = TextPreprocessor(max_vocab_size=10000, max_length=100)
    preprocessor.build_vocab(train_texts)
    test_padded, test_lengths = preprocessor.transform(test_texts)

    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(
        torch.LongTensor(test_padded),
        torch.LongTensor(test_lengths),
        torch.LongTensor(test_labels)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds, all_probas = [], []
    with torch.no_grad():
        for seqs, lengths, _ in test_loader:
            seqs, lengths = seqs.to(device), lengths.to(device)
            logits = model(seqs, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probas.extend(probs)

    return {
        'name': 'BiLSTM',
        'short': 'BiLSTM',
        'test_texts': test_texts,
        'test_true': test_labels,
        'test_preds': np.array(all_preds),
        'test_probas': np.array(all_probas)
    }


def load_distilbert(model_path='models/best_distilbert.pth', data_dir='data/processed'):
    """Load and evaluate the DistilBERT model."""
    if not os.path.exists(model_path):
        print(f"  ⚠️  DistilBERT model not found at {model_path}")
        print("     Run: python src/train_distilbert.py (use Colab with GPU)")
        return None

    try:
        from transformers import DistilBertTokenizer
        from src.models.distilbert_model import DistilBertEmotionClassifier, EmotionDatasetBERT
        from torch.utils.data import DataLoader
    except ImportError:
        print("  ⚠️  transformers library not installed. Skipping DistilBERT.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})

    model = DistilBertEmotionClassifier(
        num_classes=config.get('num_classes', 6),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    _, _, _, _, test_texts, test_labels = load_emotion_data(data_dir)

    test_dataset = EmotionDatasetBERT(
        test_texts, test_labels, tokenizer,
        max_length=config.get('max_length', 128)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds, all_probas = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probas.extend(probs)

    return {
        'name': 'DistilBERT',
        'short': 'DistilBERT',
        'test_texts': test_texts,
        'test_true': test_labels,
        'test_preds': np.array(all_preds),
        'test_probas': np.array(all_probas)
    }


# ─────────────────────────────────────────────
# Evaluation & Output
# ─────────────────────────────────────────────

def compute_metrics(result):
    """Compute accuracy, F1, and classification report for a model result."""
    true = result['test_true']
    preds = result['test_preds']
    return {
        'accuracy': float(accuracy_score(true, preds)),
        'f1_macro': float(f1_score(true, preds, average='macro')),
        'f1_weighted': float(f1_score(true, preds, average='weighted')),
        'classification_report': classification_report(
            true, preds, target_names=EMOTION_NAMES
        ),
        'confusion_matrix': confusion_matrix(true, preds).tolist()
    }


def save_combined_predictions(results, out_path='results/predictions.csv'):
    """Save all model predictions side-by-side in one CSV for Member 3."""
    if not results:
        return

    # Use the first valid result's texts/true labels as base
    base = results[0]
    df = pd.DataFrame({
        'text': base['test_texts'],
        'true_label': base['test_true'],
        'true_emotion': [EMOTION_NAMES[i] for i in base['test_true']]
    })

    for r in results:
        short = r['short'].lower().replace(' ', '_')
        df[f'pred_{short}'] = r['test_preds']
        df[f'pred_{short}_emotion'] = [EMOTION_NAMES[i] for i in r['test_preds']]
        df[f'correct_{short}'] = (r['test_true'] == r['test_preds'])
        # Add probability columns for each emotion
        for j, emotion in enumerate(EMOTION_NAMES):
            df[f'prob_{short}_{emotion}'] = r['test_probas'][:, j]

    df.to_csv(out_path, index=False)
    print(f"✅ Combined predictions → {out_path}")
    return df


def plot_confusion_matrices(results, out_path='results/confusion_matrices.png'):
    """Plot side-by-side confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold', y=1.02)

    for ax, r in zip(axes, results):
        cm = confusion_matrix(r['test_true'], r['test_preds'])
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        sns.heatmap(
            cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
            ax=ax, cbar=False, linewidths=0.5
        )
        ax.set_title(r['name'], fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrices → {out_path}")


def print_metrics_table(all_metrics):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 65)
    print("MODEL COMPARISON — TEST SET RESULTS")
    print("=" * 65)
    print(f"{'Model':<22} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}")
    print("-" * 65)
    for model_name, metrics in all_metrics.items():
        print(
            f"{model_name:<22} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['f1_macro']:>10.4f} "
            f"{metrics['f1_weighted']:>12.4f}"
        )
    print("=" * 65)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(data_dir='data/processed'):
    print("=" * 60)
    print("EVALUATING ALL MODELS")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)

    # ── Load & evaluate each model ────────────────────────────
    print("\n[1/4] Loading models and generating predictions...")

    results = []

    print("\n  → Baseline (TF-IDF + LogReg)")
    r = load_baseline(data_dir=data_dir)
    if r:
        results.append(r)

    print("\n  → BiLSTM")
    r = load_bilstm(data_dir=data_dir)
    if r:
        results.append(r)

    print("\n  → DistilBERT")
    r = load_distilbert(data_dir=data_dir)
    if r:
        results.append(r)

    if not results:
        print("\n❌ No trained models found. Train models first.")
        return

    # ── Compute metrics ───────────────────────────────────────
    print("\n[2/4] Computing metrics...")
    all_metrics = {}
    for r in results:
        metrics = compute_metrics(r)
        all_metrics[r['name']] = metrics
        print(f"\n  {r['name']}:")
        print(f"    Accuracy   : {metrics['accuracy']:.4f}")
        print(f"    F1 Macro   : {metrics['f1_macro']:.4f}")
        print(f"    F1 Weighted: {metrics['f1_weighted']:.4f}")

    print_metrics_table(all_metrics)

    # ── Save combined predictions ──────────────────────────────
    print("\n[3/4] Saving predictions...")
    save_combined_predictions(results)

    # ── Save metrics JSON ──────────────────────────────────────
    metrics_out = {
        'timestamp': datetime.now().isoformat(),
        'emotion_classes': EMOTION_NAMES,
        'models': {
            name: {k: v for k, v in m.items() if k != 'classification_report'}
            for name, m in all_metrics.items()
        }
    }
    with open('results/metrics_summary.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print("✅ Metrics summary → results/metrics_summary.json")

    # ── Confusion matrices ─────────────────────────────────────
    print("\n[4/4] Generating confusion matrices...")
    plot_confusion_matrices(results)

    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  ✅ results/predictions.csv")
    print("  ✅ results/metrics_summary.json")
    print("  ✅ results/confusion_matrices.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/processed')
    args = parser.parse_args()
    main(data_dir=args.data_dir)
