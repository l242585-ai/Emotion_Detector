"""
Models module for emotion classification.
Contains baseline (TF-IDF + Logistic Regression) and deep learning (BiLSTM) models.
"""

from .baseline_model import BaselineModel
from .deep_model import BiLSTMEmotionClassifier

__all__ = ['BaselineModel', 'BiLSTMEmotionClassifier']
