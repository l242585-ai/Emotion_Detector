"""
Complete BiLSTM Training Script — Member 2
Includes: validation loop, early stopping, checkpointing, tqdm progress, metric logging.

Usage:
    python src/train_deep.py                    # train with default hyperparams
    python src/train_deep.py --lr 0.001 --batch 32 --dropout 0.3   # custom hyperparams
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.deep_model import BiLSTMEmotionClassifier
from src.utils.data_loader import load_emotion_data, calculate_class_weights
from src.utils.preprocessing import TextPreprocessor

EMOTION_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class EmotionDataset(Dataset):
    """PyTorch Dataset for sequence-based emotion classification."""

    def __init__(self, sequences, lengths, labels):
        self.sequences = torch.LongTensor(sequences)
        self.lengths = torch.LongTensor(lengths)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for sequences, lengths, labels in tqdm(loader, desc='  Train', leave=False):
        sequences, lengths, labels = (
            sequences.to(device), lengths.to(device), labels.to(device)
        )
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping — prevents exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, lengths, labels in tqdm(loader, desc='  Eval ', leave=False):
            sequences, lengths, labels = (
                sequences.to(device), lengths.to(device), labels.to(device)
            )
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────

def train_bilstm(
    lr=0.001,
    batch_size=32,
    dropout=0.3,
    num_epochs=20,
    hidden_dim=64,
    num_layers=2,
    embedding_dim=128,
    vocab_size=10000,
    max_length=100,
    patience=5,
    save_path='models/best_bilstm.pth',
    log_to_csv=None,
    experiment_name=None,
    data_dir='data/processed'
):
    """
    Full training pipeline for BiLSTM.

    Args:
        lr (float): Learning rate
        batch_size (int): Batch size
        dropout (float): Dropout rate
        num_epochs (int): Max training epochs
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        embedding_dim (int): Word embedding dimension
        vocab_size (int): Maximum vocabulary size
        max_length (int): Max sequence length
        patience (int): Early stopping patience
        save_path (str): Where to save best model weights
        log_to_csv (str): If provided, append experiment results to this CSV
        experiment_name (str): Label for this experiment in CSV
        data_dir (str): Path to processed data directory

    Returns:
        dict: Results dictionary with metrics and history
    """
    print("=" * 60)
    print("BiLSTM EMOTION CLASSIFIER — TRAINING")
    print("=" * 60)
    print(f"  LR={lr} | Batch={batch_size} | Dropout={dropout}")
    print(f"  Layers={num_layers} | Hidden={hidden_dim} | Embedding={embedding_dim}")
    print(f"  Epochs={num_epochs} | Early Stop patience={patience}")
    print("-" * 60)

    # ── Directories ──────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)

    # ── Device ───────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # ── Load data ────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_emotion_data(data_dir)

    weight_dict, class_weights = calculate_class_weights(train_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # ── Preprocess ───────────────────────────────────────────
    print("\n[2/5] Preprocessing text...")
    preprocessor = TextPreprocessor(max_vocab_size=vocab_size, max_length=max_length)
    preprocessor.build_vocab(train_texts)

    train_padded, train_lengths = preprocessor.transform(train_texts)
    val_padded, val_lengths = preprocessor.transform(val_texts)
    test_padded, test_lengths = preprocessor.transform(test_texts)

    # ── DataLoaders ──────────────────────────────────────────
    print("\n[3/5] Creating data loaders...")
    train_loader = DataLoader(
        EmotionDataset(train_padded, train_lengths, train_labels),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        EmotionDataset(val_padded, val_lengths, val_labels),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        EmotionDataset(test_padded, test_lengths, test_labels),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ── Model ────────────────────────────────────────────────
    print("\n[4/5] Initializing model...")
    model = BiLSTMEmotionClassifier(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=6,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Learning rate scheduler — reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=False
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {total_params:,}")

    # ── Training loop ─────────────────────────────────────────
    print("\n[5/5] Training...")
    print("-" * 60)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # LR scheduler step
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Early stopping + checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': {
                    'vocab_size': preprocessor.vocab_size,
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'num_classes': 6
                }
            }, save_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"\n  ⏹  Early stopping triggered at epoch {epoch}")
                break

    # ── Load best model and evaluate on test set ──────────────
    print(f"\nLoading best model from epoch {best_epoch}...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_true = evaluate(
        model, test_loader, criterion, device
    )

    f1_macro = f1_score(test_true, test_preds, average='macro')
    f1_weighted = f1_score(test_true, test_preds, average='weighted')

    print("\n" + "=" * 60)
    print("TEST RESULTS — BiLSTM")
    print("=" * 60)
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test F1 Macro : {f1_macro:.4f}")
    print(f"  Test F1 Weighted: {f1_weighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=EMOTION_NAMES))

    # ── Save predictions ──────────────────────────────────────
    pred_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_true,
        'predicted_label': test_preds,
        'true_emotion': [EMOTION_NAMES[i] for i in test_true],
        'predicted_emotion': [EMOTION_NAMES[i] for i in test_preds],
        'correct': (test_true == test_preds)
    })
    pred_df.to_csv('results/bilstm_predictions.csv', index=False)
    print("✅ Predictions saved → results/bilstm_predictions.csv")

    # ── Save training history ──────────────────────────────────
    with open('results/bilstm_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("✅ History saved → results/bilstm_history.json")

    # ── Save metrics ───────────────────────────────────────────
    metrics = {
        'model': 'BiLSTM',
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'lr': lr, 'batch_size': batch_size, 'dropout': dropout,
            'hidden_dim': hidden_dim, 'num_layers': num_layers,
            'embedding_dim': embedding_dim
        },
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test': {
            'accuracy': float(test_acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'loss': float(test_loss)
        }
    }
    with open('results/bilstm_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Metrics saved → results/bilstm_metrics.json")

    # ── Log to CSV ─────────────────────────────────────────────
    if log_to_csv:
        log_experiment(
            csv_path=log_to_csv,
            experiment_name=experiment_name or 'bilstm_final',
            model_type='BiLSTM',
            lr=lr,
            batch_size=batch_size,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            best_epoch=best_epoch,
            val_acc=best_val_acc,
            val_loss=best_val_loss,
            test_acc=test_acc,
            f1_macro=f1_macro
        )

    results = {
        'history': history,
        'metrics': metrics,
        'test_preds': test_preds,
        'test_true': test_true
    }
    return results


def log_experiment(csv_path, experiment_name, model_type, lr, batch_size,
                   dropout, hidden_dim, num_layers, best_epoch,
                   val_acc, val_loss, test_acc, f1_macro):
    """Append one experiment row to the hyperparameter log CSV."""
    row = {
        'experiment': experiment_name,
        'model': model_type,
        'lr': lr,
        'batch_size': batch_size,
        'dropout': dropout,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'best_epoch': best_epoch,
        'val_acc': round(val_acc, 4),
        'val_loss': round(val_loss, 4),
        'test_acc': round(test_acc, 4),
        'f1_macro': round(f1_macro, 4),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    print(f"✅ Experiment logged → {csv_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train BiLSTM Emotion Classifier')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=20, help='Max epochs')
    parser.add_argument('--hidden', type=int, default=64, help='LSTM hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='LSTM layers')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to processed data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_bilstm(
        lr=args.lr,
        batch_size=args.batch,
        dropout=args.dropout,
        num_epochs=args.epochs,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        patience=args.patience,
        save_path='models/best_bilstm.pth',
        log_to_csv='experiments/hyperparameter_log.csv',
        experiment_name='bilstm_final',
        data_dir=args.data_dir
    )
