"""
Models module for emotion classification.
Contains:
  - BaselineModel: TF-IDF + Logistic Regression (Member 1)
  - BiLSTMEmotionClassifier: Bidirectional LSTM (Member 1)
  - DistilBertEmotionClassifier: Fine-tuned DistilBERT transformer (Member 2)
"""

from .baseline_model import BaselineModel
from .deep_model import BiLSTMEmotionClassifier

# DistilBERT requires the transformers library — import conditionally
try:
    from .distilbert_model import DistilBertEmotionClassifier, EmotionDatasetBERT
    _has_distilbert = True
except ImportError:
    _has_distilbert = False

__all__ = [
    'BaselineModel',
    'BiLSTMEmotionClassifier',
    'DistilBertEmotionClassifier',
    'EmotionDatasetBERT',
]
