# Model Architecture Documentation

## Overview
This document describes the two model architectures implemented for emotion classification:
1. **Baseline Model**: TF-IDF + Logistic Regression
2. **Deep Learning Model**: Bidirectional LSTM (BiLSTM)

---

## 1. Baseline Model: TF-IDF + Logistic Regression

### Architecture
- **Feature Extraction**: TF-IDF Vectorization
  - Max features: 5,000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Stop words: English
  
- **Classifier**: Multinomial Logistic Regression
  - Solver: LBFGS
  - Regularization: L2 (default)
  - Class weighting: Balanced (handles class imbalance)
  - Max iterations: 1,000

### Why This Baseline?
- **Simple and interpretable**: Can inspect feature weights
- **Fast training**: Trains in seconds on CPU
- **Strong benchmark**: TF-IDF + LogReg is competitive for text classification
- **Class balancing**: Handles imbalanced emotions (joy/sadness > surprise)

### Parameters
- Total features: ~5,000
- Trainable parameters: 5,000 × 6 = 30,000

---

## 2. Deep Learning Model: Bidirectional LSTM

### Architecture Overview

```
Input (batch_size, seq_length)
    ↓
Embedding Layer (vocab_size → 128)
    ↓
BiLSTM Layer 1 (128 → 64×2)
    ↓
BiLSTM Layer 2 (128 → 64×2)
    ↓
Concatenate [forward + backward] → 128
    ↓
Dropout (p=0.3)
    ↓
Fully Connected (128 → 6)
    ↓
Output: Logits for 6 emotions
```

### Layer Specifications

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| **Embedding** | (B, L) | (B, L, 128) | ~1.28M |
| **BiLSTM-1** | (B, L, 128) | (B, L, 128) | ~132K |
| **BiLSTM-2** | (B, L, 128) | (B, L, 128) | ~132K |
| **Concat** | (B, L, 128) | (B, 128) | 0 |
| **Dropout** | (B, 128) | (B, 128) | 0 |
| **Linear** | (B, 128) | (B, 6) | 774 |
| **Total** | | | **≈1.5M params** |

*B = batch_size, L = sequence_length (max 100)*

### Hyperparameters (Default)
- **Vocabulary size**: ~10,000 words
- **Embedding dimension**: 128
- **LSTM hidden dimension**: 64 (×2 for bidirectional = 128)
- **Number of LSTM layers**: 2
- **Dropout rate**: 0.3
- **Sequence length**: 100 tokens (padded)
- **Loss function**: CrossEntropyLoss with class weights
- **Optimizer**: Adam (learning rate: 0.001)

---

## Design Rationale

### Why BiLSTM over Standard LSTM?
- **Bidirectional context**: Captures information from both past and future
- **Better for sentiment/emotion**: Phrases like "not happy" need backward context
- **Performance**: Typically 2-5% better F1 than unidirectional LSTM

### Why 2 Layers?
- **Balance**: Single layer may underfit; 3+ layers risk overfitting on 16K samples
- **Standard practice**: 2 layers is common for moderate-sized datasets
- **Capacity**: Sufficient to learn emotion patterns without excessive parameters

### Why Embedding Dim = 128?
- **Standard choice**: Common for vocab sizes 5K-20K
- **Not too large**: 128 prevents overfitting compared to 256 or 300
- **Efficient**: Reduces parameters while maintaining expressiveness

### Why Hidden Dim = 64?
- **Output dimension**: Bidirectional gives 128 (64×2), matches embedding size
- **Parameter efficiency**: Keeps model size reasonable (~1.5M params)
- **Tunable**: Member 2 can experiment with 32, 64, 128

### Why Dropout = 0.3?
- **Regularization**: Prevents overfitting to training data
- **Common range**: 0.2-0.5 is standard; 0.3 is middle ground
- **Tunable**: Member 2 will test 0.2, 0.3, 0.5 during hyperparameter search

### Why Class Weights?
- **Imbalanced dataset**: Joy (28%) and sadness (24%) dominate; surprise is only 4%
- **Fair learning**: Class weights penalize errors on rare classes more heavily
- **Better macro-F1**: Improves performance on minority classes

### Why Max Length = 100?
- **Data analysis**: 95th percentile of text lengths is ~80 words
- **Computational efficiency**: Shorter than 512 (BERT default)
- **Sufficient coverage**: Captures most texts without excessive padding

---

## Training Strategy

### Loss Function
```python
CrossEntropyLoss(weight=class_weights)
```
- Handles multi-class classification
- Weighted to address class imbalance

### Optimizer
```python
Adam(lr=0.001, betas=(0.9, 0.999))
```
- Adaptive learning rate
- Well-suited for NLP tasks
- Learning rate to be tuned by Member 2

### Regularization
1. **Dropout (0.3)**: After LSTM layers and before FC layer
2. **Class weights**: Implicit regularization via balanced loss
3. **Early stopping**: Based on validation loss (Member 2 implements)

---

## Model Comparison

| Aspect | Baseline (TF-IDF + LogReg) | BiLSTM |
|--------|---------------------------|---------|
| **Parameters** | 30K | 1.5M |
| **Training Time** | Seconds (CPU) | Minutes (GPU) |
| **Context Awareness** | Bag of words (no order) | Sequential (order matters) |
| **Negation Handling** | Limited (via bigrams) | Good (LSTM memory) |
| **Interpretability** | High (feature weights) | Low (black box) |
| **Expected F1** | 0.65-0.75 | 0.75-0.85 |

---

## Implementation Files

- `src/models/baseline_model.py` - Baseline implementation
- `src/models/deep_model.py` - BiLSTM implementation
- `src/utils/data_loader.py` - Data loading utilities
- `src/utils/preprocessing.py` - Tokenization and vocabulary
- `src/train_baseline.py` - Baseline training script
- `src/train_deep.py` - Deep model training template (for Member 2)

---

## For Member 2: Hyperparameters to Tune

Member 2 should experiment with these hyperparameters:

| Hyperparameter | Default | Values to Test |
|----------------|---------|----------------|
| Learning rate | 0.001 | [0.0001, 0.001, 0.01] |
| Batch size | 32 | [16, 32, 64] |
| Dropout | 0.3 | [0.2, 0.3, 0.5] |
| Hidden dim | 64 | [32, 64, 128] |
| Embedding dim | 128 | [64, 128, 256] |
| Num layers | 2 | [1, 2, 3] |

**Priority**: Focus on learning rate, batch size, and dropout first.

---

## References

- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
- Schuster & Paliwal (1997). "Bidirectional Recurrent Neural Networks"
- Saravia et al. (2018). "CARER: Contextualized Affect Representations for Emotion Recognition"
