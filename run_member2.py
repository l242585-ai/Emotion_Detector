"""
Member 2 — One-Click Runner
Runs the complete Member 2 pipeline in the correct order:
  Step 1: Train Baseline (TF-IDF + LogReg)
  Step 2: Hyperparameter tuning for BiLSTM
  Step 3: Train final BiLSTM with best hyperparams
  Step 4: Hyperparameter tuning for DistilBERT (Colab/GPU)
  Step 5: Train final DistilBERT with best hyperparams
  Step 6: Evaluate all 3 models
  Step 7: Generate all visualizations

Usage:
    # Full pipeline (run in Colab for DistilBERT):
    python run_member2.py

    # Skip DistilBERT (CPU only):
    python run_member2.py --skip-distilbert

    # Only run evaluation + plots (models already trained):
    python run_member2.py --eval-only

    # Quick test to verify setup (no training):
    python run_member2.py --test-only
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))


def print_header(step, title):
    print("\n" + "=" * 65)
    print(f"  STEP {step}: {title}")
    print("=" * 65)


def check_environment():
    """Check GPU availability and installed packages."""
    print("\n" + "─" * 65)
    print("  ENVIRONMENT CHECK")
    print("─" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  Device          : {device}")
    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️  No GPU detected. DistilBERT training will be very slow.")
        print("     → Recommend using Google Colab for DistilBERT.")

    # Check transformers
    try:
        import transformers
        print(f"  Transformers    : {transformers.__version__} ✅")
        has_transformers = True
    except ImportError:
        print("  Transformers    : NOT INSTALLED ⚠️")
        print("     Install with: pip install transformers")
        has_transformers = False

    # Check seaborn
    try:
        import seaborn
        print(f"  Seaborn         : {seaborn.__version__} ✅")
    except ImportError:
        print("  Seaborn         : NOT INSTALLED (install: pip install seaborn)")

    print("─" * 65)
    return device.type, has_transformers


def test_data_loading():
    """Quick test to verify data loads correctly."""
    print("\n  Testing data loading...")
    from src.utils.data_loader import load_emotion_data
    try:
        train, train_l, val, val_l, test, test_l = load_emotion_data()
        print(f"  ✅ Data loaded: {len(train)} train | {len(val)} val | {len(test)} test")
        return True
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False


def test_model_forward_passes():
    """Quick forward pass test for all models."""
    print("\n  Testing model forward passes...")

    # Baseline
    try:
        from src.models.baseline_model import BaselineModel
        m = BaselineModel(max_features=100)
        m.fit(['I am happy', 'I am sad', 'feeling great', 'terrible day',
               'love this', 'angry now'], [0, 1, 0, 1, 0, 1])
        pred = m.predict(['test text'])
        print(f"  ✅ Baseline forward pass OK")
    except Exception as e:
        print(f"  ❌ Baseline failed: {e}")

    # BiLSTM
    try:
        import torch
        from src.models.deep_model import BiLSTMEmotionClassifier
        m = BiLSTMEmotionClassifier(vocab_size=100, num_classes=6)
        x = torch.randint(0, 100, (2, 10))
        y = m(x)
        assert y.shape == (2, 6)
        print(f"  ✅ BiLSTM forward pass OK (output shape: {y.shape})")
    except Exception as e:
        print(f"  ❌ BiLSTM failed: {e}")

    # DistilBERT
    try:
        from src.models.distilbert_model import DistilBertEmotionClassifier
        print("  ✅ DistilBERT import OK (full forward pass skipped in test mode)")
    except ImportError as e:
        print(f"  ⚠️  DistilBERT import failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Member 2 — Full Training Pipeline')
    parser.add_argument('--skip-distilbert', action='store_true',
                        help='Skip DistilBERT training (CPU-only mode)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation and visualization (models must be pre-trained)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test environment setup, no training')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip hyperparameter tuning, train directly with defaults')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to processed data')
    args = parser.parse_args()

    print("\n" + "█" * 65)
    print("  EMOTION DETECTOR — MEMBER 2 PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 65)

    # ── Environment check ─────────────────────────────────────
    device_type, has_transformers = check_environment()

    if args.test_only:
        print("\n  Running setup tests...")
        ok = test_data_loading()
        if ok:
            test_model_forward_passes()
        print("\n  ✅ Test complete! If all checks passed, you're ready to train.")
        print("  Run: python run_member2.py")
        return

    # ── Create directories ────────────────────────────────────
    for d in ['models', 'models/checkpoints', 'results', 'experiments']:
        os.makedirs(d, exist_ok=True)

    if not args.eval_only:
        # ══════════════════════════════════════════════════════
        # STEP 1: Baseline
        # ══════════════════════════════════════════════════════
        print_header(1, "TRAIN BASELINE (TF-IDF + Logistic Regression)")
        if os.path.exists('models/best_baseline.pkl'):
            print("  ⏭️  Baseline already trained. Skipping.")
        else:
            os.system(f'python src/train_baseline.py')

        # ══════════════════════════════════════════════════════
        # STEP 2 & 3: BiLSTM Tuning + Final Training
        # ══════════════════════════════════════════════════════
        print_header(2, "BILSTM HYPERPARAMETER TUNING")

        best_bilstm_lr = 0.001
        best_bilstm_batch = 32
        best_bilstm_dropout = 0.3

        if not args.skip_tuning:
            from src.hyperparameter_tuning import tune_bilstm
            best_config = tune_bilstm(data_dir=args.data_dir)
            best_bilstm_lr = best_config['lr']
            best_bilstm_batch = best_config['batch_size']
            best_bilstm_dropout = best_config['dropout']
        else:
            print("  ⏭️  Skipping tuning (--skip-tuning flag). Using defaults.")

        print_header(3, "TRAIN FINAL BILSTM (Best Hyperparameters)")
        from src.train_deep import train_bilstm
        train_bilstm(
            lr=best_bilstm_lr,
            batch_size=best_bilstm_batch,
            dropout=best_bilstm_dropout,
            num_epochs=25,
            patience=5,
            save_path='models/best_bilstm.pth',
            log_to_csv='experiments/hyperparameter_log.csv',
            experiment_name='bilstm_final_best',
            data_dir=args.data_dir
        )

        # Copy best BiLSTM as overall best deep model
        import shutil
        shutil.copy('models/best_bilstm.pth', 'models/best_model.pth')
        print("  ✅ Copied BiLSTM as models/best_model.pth")

        # ══════════════════════════════════════════════════════
        # STEP 4 & 5: DistilBERT Tuning + Final Training
        # ══════════════════════════════════════════════════════
        if args.skip_distilbert:
            print("\n  ⏭️  Skipping DistilBERT (--skip-distilbert flag)")
        elif not has_transformers:
            print("\n  ⚠️  Skipping DistilBERT (transformers not installed)")
            print("     Install: pip install transformers && python run_member2.py")
        else:
            print_header(4, "DISTILBERT HYPERPARAMETER TUNING")

            best_bert_lr = 2e-5
            best_bert_batch = 16
            best_bert_dropout = 0.3

            if not args.skip_tuning:
                from src.hyperparameter_tuning import tune_distilbert
                best_config = tune_distilbert(data_dir=args.data_dir)
                best_bert_lr = best_config['lr']
                best_bert_batch = best_config['batch_size']
                best_bert_dropout = best_config['dropout']
            else:
                print("  ⏭️  Skipping tuning. Using defaults.")

            print_header(5, "TRAIN FINAL DISTILBERT (Best Hyperparameters)")
            from src.train_distilbert import train_distilbert
            train_distilbert(
                lr=best_bert_lr,
                batch_size=best_bert_batch,
                dropout=best_bert_dropout,
                num_epochs=5,
                patience=3,
                save_path='models/best_distilbert.pth',
                log_to_csv='experiments/hyperparameter_log.csv',
                experiment_name='distilbert_final_best',
                data_dir=args.data_dir
            )

            # Update best_model.pth to DistilBERT if it outperforms BiLSTM
            _maybe_update_best_model()

    # ══════════════════════════════════════════════════════
    # STEP 6: Evaluate All Models
    # ══════════════════════════════════════════════════════
    print_header(6, "EVALUATE ALL MODELS")
    from src.evaluate_all import main as evaluate_main
    evaluate_main(data_dir=args.data_dir)

    # ══════════════════════════════════════════════════════
    # STEP 7: Visualizations
    # ══════════════════════════════════════════════════════
    print_header(7, "GENERATE VISUALIZATIONS")
    from src.visualize_results import main as visualize_main
    visualize_main()

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "█" * 65)
    print("  ✅ MEMBER 2 PIPELINE COMPLETE!")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 65)
    print("\n📦 DELIVERABLES FOR MEMBER 3:")
    _print_deliverables()


def _maybe_update_best_model():
    """Compare BiLSTM and DistilBERT metrics and update best_model.pth."""
    import shutil

    bilstm_path = 'results/bilstm_metrics.json'
    bert_path = 'results/distilbert_metrics.json'

    if not (os.path.exists(bilstm_path) and os.path.exists(bert_path)):
        return

    with open(bilstm_path) as f:
        bilstm_f1 = json.load(f)['test']['f1_macro']
    with open(bert_path) as f:
        bert_f1 = json.load(f)['test']['f1_macro']

    if bert_f1 > bilstm_f1:
        shutil.copy('models/best_distilbert.pth', 'models/best_model.pth')
        print(f"  ✅ DistilBERT is best model (F1={bert_f1:.4f} > BiLSTM F1={bilstm_f1:.4f})")
        print("     Copied to models/best_model.pth")
    else:
        print(f"  ✅ BiLSTM remains best model (F1={bilstm_f1:.4f} >= DistilBERT F1={bert_f1:.4f})")


def _print_deliverables():
    deliverables = [
        ('models/best_model.pth',               'Best trained model weights'),
        ('models/best_bilstm.pth',              'BiLSTM weights'),
        ('models/best_distilbert.pth',          'DistilBERT weights (if trained)'),
        ('results/predictions.csv',             'All model predictions on test set'),
        ('results/metrics_summary.json',        'Accuracy + F1 for all models'),
        ('experiments/hyperparameter_log.csv',  'All tuning experiments'),
        ('results/training_curves.png',         'Loss/accuracy training curves'),
        ('results/model_comparison.png',        'Model comparison bar chart'),
        ('results/confusion_matrices.png',      'Confusion matrices'),
        ('results/per_class_f1.png',            'Per-class F1 comparison'),
    ]

    for path, desc in deliverables:
        status = "✅" if os.path.exists(path) else "⬜"
        print(f"  {status} {path:<45} {desc}")


if __name__ == '__main__':
    main()
