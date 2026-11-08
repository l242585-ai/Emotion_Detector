"""
DistilBERT Training Script — Member 2
Fine-tunes distilbert-base-uncased for 6-class emotion classification.
Includes: validation loop, early stopping, checkpointing, tqdm, metric logging.

Usage (Colab / VS Code with GPU):
    python src/train_distilbert.py
    python src/train_distilbert.py --lr 2e-5 --batch 16 --dropout 0.3
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.distilbert_model import DistilBertEmotionClassifier, EmotionDatasetBERT
from src.utils.data_loader import load_emotion_data, calculate_class_weights

# Try importing transformers (required for DistilBERT)
try:
    from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
except ImportError:
    print("ERROR: 'transformers' library not installed.")
    print("Run: pip install transformers")
    sys.exit(1)

EMOTION_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Run one fine-tuning epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc='  Train', leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — essential for transformer fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate DistilBERT model. Returns (avg_loss, accuracy, preds, true_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='  Eval ', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        correct / total,
        np.array(all_preds),
        np.array(all_labels)
    )


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────

def train_distilbert(
    lr=2e-5,
    batch_size=16,
    dropout=0.3,
    num_epochs=5,
    max_length=128,
    patience=3,
    save_path='models/best_distilbert.pth',
    log_to_csv=None,
    experiment_name=None,
    data_dir='data/processed'
):
    """
    Full fine-tuning pipeline for DistilBERT.

    Args:
        lr (float): Learning rate (recommended: 1e-5 to 5e-5 for transformers)
        batch_size (int): Batch size (16 or 32 for transformers)
        dropout (float): Dropout on classifier head
        num_epochs (int): Max training epochs (3-5 is usually enough for BERT)
        max_length (int): Max token length (128 is standard)
        patience (int): Early stopping patience
        save_path (str): Path to save best model
        log_to_csv (str): If provided, log results to this CSV
        experiment_name (str): Label for CSV logging
        data_dir (str): Path to processed data

    Returns:
        dict: Results dictionary
    """
    print("=" * 60)
    print("DistilBERT EMOTION CLASSIFIER — FINE-TUNING")
    print("=" * 60)
    print(f"  LR={lr} | Batch={batch_size} | Dropout={dropout}")
    print(f"  Epochs={num_epochs} | MaxLen={max_length} | Patience={patience}")
    print("-" * 60)

    # ── Directories ──────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)

    # ── Device ───────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cpu':
        print("  ⚠️  WARNING: DistilBERT on CPU is very slow.")
        print("     Strongly recommend using Google Colab with GPU runtime.")

    # ── Load data ────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_emotion_data(data_dir)

    weight_dict, class_weights = calculate_class_weights(train_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # ── Tokenizer ────────────────────────────────────────────
    print("\n[2/5] Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # ── Datasets & DataLoaders ───────────────────────────────
    print("\n[3/5] Creating data loaders...")
    train_dataset = EmotionDatasetBERT(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmotionDatasetBERT(val_texts, val_labels, tokenizer, max_length)
    test_dataset = EmotionDatasetBERT(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  ✓ Train batches : {len(train_loader)}")
    print(f"  ✓ Val batches   : {len(val_loader)}")
    print(f"  ✓ Test batches  : {len(test_loader)}")

    # ── Model ────────────────────────────────────────────────
    print("\n[4/5] Loading pretrained DistilBERT...")
    model = DistilBertEmotionClassifier(
        num_classes=6,
        dropout=dropout,
        pretrained_model='distilbert-base-uncased'
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer: use AdamW (recommended for transformers — applies weight decay correctly)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup scheduler — standard for BERT fine-tuning
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters    : {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")

    # ── Training loop ─────────────────────────────────────────
    print("\n[5/5] Fine-tuning DistilBERT...")
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

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

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
                    'num_classes': 6,
                    'dropout': dropout,
                    'max_length': max_length
                }
            }, save_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"\n  ⏹  Early stopping triggered at epoch {epoch}")
                break

    # ── Load best and evaluate test set ───────────────────────
    print(f"\nLoading best model from epoch {best_epoch}...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_true = evaluate(
        model, test_loader, criterion, device
    )

    f1_macro = f1_score(test_true, test_preds, average='macro')
    f1_weighted = f1_score(test_true, test_preds, average='weighted')

    print("\n" + "=" * 60)
    print("TEST RESULTS — DistilBERT")
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
    pred_df.to_csv('results/distilbert_predictions.csv', index=False)
    print("✅ Predictions saved → results/distilbert_predictions.csv")

    # ── Save history ──────────────────────────────────────────
    with open('results/distilbert_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("✅ History saved → results/distilbert_history.json")

    # ── Save metrics ───────────────────────────────────────────
    metrics = {
        'model': 'DistilBERT',
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'lr': lr, 'batch_size': batch_size,
            'dropout': dropout, 'max_length': max_length
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
    with open('results/distilbert_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Metrics saved → results/distilbert_metrics.json")

    # ── Log to CSV ─────────────────────────────────────────────
    if log_to_csv:
        _log_experiment(
            csv_path=log_to_csv,
            experiment_name=experiment_name or 'distilbert_final',
            model_type='DistilBERT',
            lr=lr, batch_size=batch_size, dropout=dropout,
            hidden_dim=768, num_layers=6,
            best_epoch=best_epoch,
            val_acc=best_val_acc, val_loss=best_val_loss,
            test_acc=test_acc, f1_macro=f1_macro
        )

    return {
        'history': history,
        'metrics': metrics,
        'test_preds': test_preds,
        'test_true': test_true
    }


def _log_experiment(csv_path, experiment_name, model_type, lr, batch_size,
                    dropout, hidden_dim, num_layers, best_epoch,
                    val_acc, val_loss, test_acc, f1_macro):
    """Append experiment to hyperparameter log CSV."""
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


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT for Emotion Classification')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=5, help='Max epochs (default: 5)')
    parser.add_argument('--max-len', type=int, default=128, help='Max token length (default: 128)')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (default: 3)')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_distilbert(
        lr=args.lr,
        batch_size=args.batch,
        dropout=args.dropout,
        num_epochs=args.epochs,
        max_length=args.max_len,
        patience=args.patience,
        save_path='models/best_distilbert.pth',
        log_to_csv='experiments/hyperparameter_log.csv',
        experiment_name='distilbert_final',
        data_dir=args.data_dir
    )
