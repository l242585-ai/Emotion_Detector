# Baseline Model Results
**Model**: TF-IDF + Logistic Regression  
**Date**: Day 2 - Member 1  

---

## Performance Summary

| Metric | Validation | Test |
|--------|-----------|------|
| **Accuracy** | 90.60% | 89.63% |
| **F1 (macro)** | 88.16% | 85.97% |
| **F1 (weighted)** | 90.72% | 89.93% |

---

## Hyperparameters

- **Max Features**: 5,000 (TF-IDF)
- **N-gram Range**: (1, 2) - unigrams + bigrams
- **Classifier**: Logistic Regression (multinomial)
- **Regularization**: L2 (default)
- **Class Weighting**: Balanced
- **Max Iterations**: 1,000

---

## Per-Class Performance (Test Set)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Sadness** | 0.96 | 0.91 | 0.94 | 572 |
| **Joy** | 0.96 | 0.88 | 0.92 | 686 |
| **Love** | 0.70 | 0.93 | 0.80 | 159 |
| **Anger** | 0.87 | 0.91 | 0.89 | 273 |
| **Fear** | 0.89 | 0.84 | 0.87 | 223 |
| **Surprise** | 0.62 | 0.94 | 0.75 | 64 |

---

## Key Observations

### Strengths
- **High overall accuracy** (89.6%) - strong baseline
- **Excellent recall** on minority classes (Love: 93%, Surprise: 94%)
- **Fast training** - trains in seconds on CPU
- **Class imbalance handled well** with balanced weights

### Weaknesses
- **Lower precision** on minority classes (Love: 70%, Surprise: 62%)
- **Surprise confusion** - high recall but low precision suggests over-prediction
- **No sequential context** - bag-of-words approach ignores word order

---

## Top Predictive Features

### Sadness
punished (3.58), melancholy (3.56), lethargic (3.56), troubled (3.25)

### Joy
pleasant (3.15), resolved (3.12), innocent (3.11), successful (3.10)

### Love
caring (6.96), loyal (6.19), sympathetic (6.18), loving (6.12)

### Anger
dangerous (5.34), greedy (5.12), fucked (5.07), bothered (5.07)

### Fear
shaken (5.98), terrified (5.78), reluctant (5.33), vulnerable (5.18)

### Surprise
amazed (9.53), impressed (9.46), surprised (9.04), curious (8.68)

---

## Class Distribution & Weights

| Emotion | Count | Percentage | Weight |
|---------|-------|------------|--------|
| Sadness | 4,557 | 29.2% | 0.570 |
| Joy | 5,233 | 33.6% | 0.497 |
| Love | 1,260 | 8.1% | 2.063 |
| Anger | 2,100 | 13.5% | 1.238 |
| Fear | 1,892 | 12.1% | 1.374 |
| Surprise | 552 | 3.5% | 4.708 |

---

## Files Generated

- ✅ `results/baseline_predictions.csv` - Test predictions with probabilities
- ✅ `results/baseline_metrics.json` - Detailed metrics in JSON format
- ✅ `models/best_baseline.pkl` - Trained model (serialized)

---

## Comparison Target for Deep Learning

The BiLSTM model should aim to beat:
- **Test Accuracy**: 89.63%
- **Macro F1**: 85.97%

Expected improvements:
- Better handling of negation ("not happy")
- Improved precision on minority classes
- Sequential context understanding
