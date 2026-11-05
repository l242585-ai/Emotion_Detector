"""
Baseline Model Training Script
Trains TF-IDF + Logistic Regression baseline for emotion classification.

Usage:
    python src/train_baseline.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.baseline_model import BaselineModel
from src.utils.data_loader import load_emotion_data, calculate_class_weights


def main():
    print("="*60)
    print("BASELINE MODEL TRAINING: TF-IDF + Logistic Regression")
    print("="*60)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_emotion_data()
    
    # Calculate class weights (for information)
    print("\n[2/5] Analyzing class distribution...")
    weight_dict, weights = calculate_class_weights(train_labels)
    
    # Initialize and train baseline model
    print("\n[3/5] Training baseline model...")
    print("-" * 60)
    
    model = BaselineModel(
        max_features=5000,
        ngram_range=(1, 2),
        max_iter=1000
    )
    
    model.fit(train_texts, train_labels)
    
    # Evaluate on validation set
    print("\n[4/5] Evaluating on validation set...")
    print("-" * 60)
    
    val_results = model.evaluate(val_texts, val_labels)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy:    {val_results['accuracy']:.4f}")
    print(f"  F1 (macro):  {val_results['f1_macro']:.4f}")
    print(f"  F1 (weighted): {val_results['f1_weighted']:.4f}")
    
    print("\nClassification Report (Validation):")
    print(val_results['classification_report'])
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    print("-" * 60)
    
    test_results = model.evaluate(test_texts, test_labels)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:    {test_results['accuracy']:.4f}")
    print(f"  F1 (macro):  {test_results['f1_macro']:.4f}")
    print(f"  F1 (weighted): {test_results['f1_weighted']:.4f}")
    
    print("\nClassification Report (Test):")
    print(test_results['classification_report'])
    
    # Generate predictions
    print("\nGenerating predictions...")
    test_predictions = model.predict(test_texts)
    test_probabilities = model.predict_proba(test_texts)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels,
        'predicted_label': test_predictions,
        'prob_sadness': test_probabilities[:, 0],
        'prob_joy': test_probabilities[:, 1],
        'prob_love': test_probabilities[:, 2],
        'prob_anger': test_probabilities[:, 3],
        'prob_fear': test_probabilities[:, 4],
        'prob_surprise': test_probabilities[:, 5],
    })
    
    predictions_df.to_csv('results/baseline_predictions.csv', index=False)
    print("✓ Predictions saved to results/baseline_predictions.csv")
    
    # Save metrics
    metrics = {
        'model': 'TF-IDF + Logistic Regression',
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'max_features': 5000,
            'ngram_range': '(1, 2)',
            'max_iter': 1000,
            'class_weight': 'balanced'
        },
        'validation': {
            'accuracy': float(val_results['accuracy']),
            'f1_macro': float(val_results['f1_macro']),
            'f1_weighted': float(val_results['f1_weighted'])
        },
        'test': {
            'accuracy': float(test_results['accuracy']),
            'f1_macro': float(test_results['f1_macro']),
            'f1_weighted': float(test_results['f1_weighted'])
        },
        'data': {
            'train_samples': len(train_texts),
            'val_samples': len(val_texts),
            'test_samples': len(test_texts)
        }
    }
    
    with open('results/baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Metrics saved to results/baseline_metrics.json")
    
    # Save model
    model.save('models/best_baseline.pkl')
    print("✓ Model saved to models/best_baseline.pkl")
    
    # Get feature importance
    print("\nTop features per emotion class:")
    print("-" * 60)
    importance = model.get_feature_importance(top_n=10)
    
    emotion_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    for class_idx, features in importance.items():
        print(f"\n{emotion_names[class_idx].upper()}:")
        for word, coef in features[:5]:
            print(f"  {word:20s} {coef:8.4f}")
    
    print("\n" + "="*60)
    print("BASELINE TRAINING COMPLETE!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Test F1 (macro): {test_results['f1_macro']:.4f}")
    print(f"\nFiles created:")
    print(f"  ✓ results/baseline_predictions.csv")
    print(f"  ✓ results/baseline_metrics.json")
    print(f"  ✓ models/best_baseline.pkl")
    print("\n")


if __name__ == '__main__':
    main()
