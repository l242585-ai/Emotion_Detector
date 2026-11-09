"""
Hyperparameter Tuning — Smart Sequential Search (Member 2)
Runs 3-phase sequential search for BiLSTM and DistilBERT:
  Phase 1: Tune learning rate  (batch=default, dropout=default fixed)
  Phase 2: Tune batch size     (best LR, dropout=default fixed)
  Phase 3: Tune dropout        (best LR + batch)

All results logged to experiments/hyperparameter_log.csv

Usage:
    python src/hyperparameter_tuning.py --model bilstm
    python src/hyperparameter_tuning.py --model distilbert
    python src/hyperparameter_tuning.py --model both
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

LOG_CSV = 'experiments/hyperparameter_log.csv'
RESULTS_DIR = 'results'

# ── Hyperparameter search spaces ──────────────────────────────────────────────
BILSTM_SEARCH = {
    'lr_values':      [0.0001, 0.001, 0.01],
    'batch_values':   [16, 32, 64],
    'dropout_values': [0.2, 0.3, 0.5],
    # Defaults held fixed during search
    'default_lr':      0.001,
    'default_batch':   32,
    'default_dropout': 0.3,
}

DISTILBERT_SEARCH = {
    'lr_values':      [1e-5, 2e-5, 5e-5],
    'batch_values':   [8, 16, 32],
    'dropout_values': [0.1, 0.2, 0.3],
    'default_lr':      2e-5,
    'default_batch':   16,
    'default_dropout': 0.3,
}


# ─────────────────────────────────────────────
# BiLSTM Tuning
# ─────────────────────────────────────────────

def tune_bilstm(data_dir='data/processed'):
    """
    Run 3-phase sequential hyperparameter search for BiLSTM.
    Returns the best configuration found.
    """
    from src.train_deep import train_bilstm

    search = BILSTM_SEARCH
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING — BiLSTM")
    print("Smart Sequential Search (3 Phases)")
    print("=" * 60)

    phase_results = {}

    # ── Phase 1: Learning Rate ────────────────────────────────
    print("\n📌 PHASE 1: Tuning Learning Rate")
    print(f"   LR values: {search['lr_values']}")
    print(f"   (batch={search['default_batch']}, dropout={search['default_dropout']} fixed)")
    print("-" * 60)

    lr_results = []
    for lr in search['lr_values']:
        exp_name = f"bilstm_lr_{lr}"
        print(f"\n  ▶ Experiment: {exp_name}")
        result = train_bilstm(
            lr=lr,
            batch_size=search['default_batch'],
            dropout=search['default_dropout'],
            num_epochs=10,         # Fewer epochs for tuning
            patience=3,
            save_path=f'models/checkpoints/{exp_name}.pth',
            log_to_csv=LOG_CSV,
            experiment_name=exp_name,
            data_dir=data_dir
        )
        lr_results.append({
            'lr': lr,
            'val_acc': result['metrics']['best_val_acc'],
            'val_loss': result['metrics']['best_val_loss'],
            'f1_macro': result['metrics']['test']['f1_macro']
        })

    best_lr_row = max(lr_results, key=lambda x: x['val_acc'])
    best_lr = best_lr_row['lr']
    print(f"\n  ✅ Best LR: {best_lr} (val_acc={best_lr_row['val_acc']:.4f})")
    phase_results['phase1_lr'] = lr_results
    phase_results['best_lr'] = best_lr

    _save_phase_results('bilstm', phase_results)

    # ── Phase 2: Batch Size ───────────────────────────────────
    print("\n📌 PHASE 2: Tuning Batch Size")
    print(f"   Batch values: {search['batch_values']}")
    print(f"   (lr={best_lr}, dropout={search['default_dropout']} fixed)")
    print("-" * 60)

    batch_results = []
    for batch in search['batch_values']:
        exp_name = f"bilstm_batch_{batch}"
        print(f"\n  ▶ Experiment: {exp_name}")
        result = train_bilstm(
            lr=best_lr,
            batch_size=batch,
            dropout=search['default_dropout'],
            num_epochs=10,
            patience=3,
            save_path=f'models/checkpoints/{exp_name}.pth',
            log_to_csv=LOG_CSV,
            experiment_name=exp_name,
            data_dir=data_dir
        )
        batch_results.append({
            'batch': batch,
            'val_acc': result['metrics']['best_val_acc'],
            'val_loss': result['metrics']['best_val_loss'],
            'f1_macro': result['metrics']['test']['f1_macro']
        })

    best_batch_row = max(batch_results, key=lambda x: x['val_acc'])
    best_batch = best_batch_row['batch']
    print(f"\n  ✅ Best Batch: {best_batch} (val_acc={best_batch_row['val_acc']:.4f})")
    phase_results['phase2_batch'] = batch_results
    phase_results['best_batch'] = best_batch

    _save_phase_results('bilstm', phase_results)

    # ── Phase 3: Dropout ──────────────────────────────────────
    print("\n📌 PHASE 3: Tuning Dropout")
    print(f"   Dropout values: {search['dropout_values']}")
    print(f"   (lr={best_lr}, batch={best_batch} fixed)")
    print("-" * 60)

    dropout_results = []
    for dropout in search['dropout_values']:
        exp_name = f"bilstm_dropout_{dropout}"
        print(f"\n  ▶ Experiment: {exp_name}")
        result = train_bilstm(
            lr=best_lr,
            batch_size=best_batch,
            dropout=dropout,
            num_epochs=10,
            patience=3,
            save_path=f'models/checkpoints/{exp_name}.pth',
            log_to_csv=LOG_CSV,
            experiment_name=exp_name,
            data_dir=data_dir
        )
        dropout_results.append({
            'dropout': dropout,
            'val_acc': result['metrics']['best_val_acc'],
            'val_loss': result['metrics']['best_val_loss'],
            'f1_macro': result['metrics']['test']['f1_macro']
        })

    best_dropout_row = max(dropout_results, key=lambda x: x['val_acc'])
    best_dropout = best_dropout_row['dropout']
    print(f"\n  ✅ Best Dropout: {best_dropout} (val_acc={best_dropout_row['val_acc']:.4f})")
    phase_results['phase3_dropout'] = dropout_results
    phase_results['best_dropout'] = best_dropout

    _save_phase_results('bilstm', phase_results)

    # ── Summary ───────────────────────────────────────────────
    best_config = {
        'model': 'BiLSTM',
        'lr': best_lr,
        'batch_size': best_batch,
        'dropout': best_dropout
    }

    print("\n" + "=" * 60)
    print("✅ BiLSTM TUNING COMPLETE — BEST CONFIGURATION")
    print("=" * 60)
    print(f"   Learning Rate : {best_lr}")
    print(f"   Batch Size    : {best_batch}")
    print(f"   Dropout       : {best_dropout}")
    print("=" * 60)

    return best_config


# ─────────────────────────────────────────────
# DistilBERT Tuning
# ─────────────────────────────────────────────

def tune_distilbert(data_dir='data/processed'):
    """
    Run 3-phase sequential hyperparameter search for DistilBERT.
    Note: DistilBERT is less sensitive to hyperparams than BiLSTM — use
    fewer epochs per experiment (2-3) to save time.
    Returns the best configuration found.
    """
    from src.train_distilbert import train_distilbert

    search = DISTILBERT_SEARCH
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING — DistilBERT")
    print("Smart Sequential Search (3 Phases)")
    print("⚠️  Requires GPU (Google Colab recommended)")
    print("=" * 60)

    phase_results = {}

    # ── Phase 1: Learning Rate ────────────────────────────────
    print("\n📌 PHASE 1: Tuning Learning Rate")
    print(f"   LR values: {search['lr_values']}")
    print("-" * 60)

    lr_results = []
    for lr in search['lr_values']:
        exp_name = f"distilbert_lr_{lr}"
        print(f"\n  ▶ Experiment: {exp_name}")
        result = train_distilbert(
            lr=lr,
            batch_size=search['default_batch'],
            dropout=search['default_dropout'],
            num_epochs=3,
            patience=2,
            save_path=f'models/checkpoints/{exp_name}.pth',
            log_to_csv=LOG_CSV,
            experiment_name=exp_name,
            data_dir=data_dir
        )
        lr_results.append({
            'lr': lr,
            'val_acc': result['metrics']['best_val_acc'],
            'val_loss': result['metrics']['best_val_loss'],
            'f1_macro': result['metrics']['test']['f1_macro']
        })

    best_lr_row = max(lr_results, key=lambda x: x['val_acc'])
    best_lr = best_lr_row['lr']
    print(f"\n  ✅ Best LR: {best_lr} (val_acc={best_lr_row['val_acc']:.4f})")
    phase_results['phase1_lr'] = lr_results
    phase_results['best_lr'] = best_lr

    _save_phase_results('distilbert', phase_results)

    # ── Phase 2: Batch Size ───────────────────────────────────
    print("\n📌 PHASE 2: Tuning Batch Size")
    print(f"   Batch values: {search['batch_values']}")
    print("-" * 60)

    batch_results = []
    for batch in search['batch_values']:
        exp_name = f"distilbert_batch_{batch}"
        print(f"\n  ▶ Experiment: {exp_name}")
        result = train_distilbert(
            lr=best_lr,
            batch_size=batch,
            dropout=search['default_dropout'],
            num_epochs=3,
            patience=2,
            save_path=f'models/checkpoints/{exp_name}.pth',
            log_to_csv=LOG_CSV,
            experiment_name=exp_name,
            data_dir=data_dir
        )
        batch_results.append({
            'batch': batch,
            'val_acc': result['metrics']['best_val_acc'],
            'val_loss': result['metrics']['best_val_loss'],
            'f1_macro': result['metrics']['test']['f1_macro']
        })

    best_batch_row = max(batch_results, key=lambda x: x['val_acc'])
    best_batch = best_batch_row['batch']
    print(f"\n  ✅ Best Batch: {best_batch} (val_acc={best_batch_row['val_acc']:.4f})")
    phase_results['phase2_batch'] = batch_results
    phase_results['best_batch'] = best_batch

    _save_phase_results('distilbert', phase_results)

    # ── Phase 3: Dropout ──────────────────────────────────────
    print("\n📌 PHASE 3: Tuning Dropout")
    print(f"   Dropout values: {search['dropout_values']}")
    print("-" * 60)

    dropout_results = []
    for dropout in search['dropout_values']:
        exp_name = f"distilbert_dropout_{dropout}"
        print(f"\n  ▶ Experiment: {exp_name}")
        result = train_distilbert(
            lr=best_lr,
            batch_size=best_batch,
            dropout=dropout,
            num_epochs=3,
            patience=2,
            save_path=f'models/checkpoints/{exp_name}.pth',
            log_to_csv=LOG_CSV,
            experiment_name=exp_name,
            data_dir=data_dir
        )
        dropout_results.append({
            'dropout': dropout,
            'val_acc': result['metrics']['best_val_acc'],
            'val_loss': result['metrics']['best_val_loss'],
            'f1_macro': result['metrics']['test']['f1_macro']
        })

    best_dropout_row = max(dropout_results, key=lambda x: x['val_acc'])
    best_dropout = best_dropout_row['dropout']
    print(f"\n  ✅ Best Dropout: {best_dropout} (val_acc={best_dropout_row['val_acc']:.4f})")
    phase_results['phase3_dropout'] = dropout_results
    phase_results['best_dropout'] = best_dropout

    _save_phase_results('distilbert', phase_results)

    best_config = {
        'model': 'DistilBERT',
        'lr': best_lr,
        'batch_size': best_batch,
        'dropout': best_dropout
    }

    print("\n" + "=" * 60)
    print("✅ DistilBERT TUNING COMPLETE — BEST CONFIGURATION")
    print("=" * 60)
    print(f"   Learning Rate : {best_lr}")
    print(f"   Batch Size    : {best_batch}")
    print(f"   Dropout       : {best_dropout}")
    print("=" * 60)

    return best_config


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _save_phase_results(model_name, phase_results):
    """Save intermediate tuning results to JSON."""
    path = f'experiments/{model_name}_tuning_progress.json'
    with open(path, 'w') as f:
        json.dump(phase_results, f, indent=2, default=str)


def print_tuning_summary():
    """Print a formatted summary table from the hyperparameter log CSV."""
    if not os.path.exists(LOG_CSV):
        print("No hyperparameter log found. Run tuning first.")
        return

    df = pd.read_csv(LOG_CSV)
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 80)

    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name].copy()
        print(f"\n📊 {model_name}:")
        print(model_df[['experiment', 'lr', 'batch_size', 'dropout',
                        'val_acc', 'f1_macro', 'best_epoch']].to_string(index=False))

    # Overall best per model
    print("\n" + "─" * 80)
    print("🏆 Best per Model:")
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        best = model_df.loc[model_df['val_acc'].idxmax()]
        print(f"  {model_name:12s}: val_acc={best['val_acc']:.4f} | "
              f"f1_macro={best['f1_macro']:.4f} | "
              f"lr={best['lr']} | batch={best['batch_size']} | dropout={best['dropout']}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for Emotion Classifier'
    )
    parser.add_argument(
        '--model', type=str, default='bilstm',
        choices=['bilstm', 'distilbert', 'both'],
        help='Which model to tune (default: bilstm)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--summary-only', action='store_true',
        help='Only print existing tuning summary, do not run tuning'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.summary_only:
        print_tuning_summary()
        sys.exit(0)

    best_configs = {}

    if args.model in ('bilstm', 'both'):
        best_bilstm = tune_bilstm(data_dir=args.data_dir)
        best_configs['bilstm'] = best_bilstm

    if args.model in ('distilbert', 'both'):
        best_distilbert = tune_distilbert(data_dir=args.data_dir)
        best_configs['distilbert'] = best_distilbert

    # Save best configs
    with open('experiments/best_configs.json', 'w') as f:
        json.dump(best_configs, f, indent=2)
    print("\n✅ Best configurations saved → experiments/best_configs.json")

    # Print summary table
    print_tuning_summary()
