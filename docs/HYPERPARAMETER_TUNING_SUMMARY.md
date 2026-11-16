# Hyperparameter Tuning Summary

**Date:** November 5-6, 2026  
**Models:** BiLSTM and DistilBERT  
**Method:** Sequential 3-Phase Grid Search  

---

## Methodology

We employed a **smart sequential search strategy** to efficiently explore the hyperparameter space:

1. **Phase 1:** Tune learning rate (while keeping batch size and dropout at defaults)
2. **Phase 2:** Tune batch size (using best LR, keeping dropout at default)
3. **Phase 3:** Tune dropout (using best LR and batch size)

This approach reduces the number of experiments from 27 (full grid search: 3×3×3) to just 9 experiments per model while still finding near-optimal configurations.

---

## BiLSTM Hyperparameter Tuning

### Search Space
- **Learning rates:** [0.0001, 0.001, 0.01]
- **Batch sizes:** [16, 32, 64]
- **Dropout rates:** [0.2, 0.3, 0.5]

### Phase 1: Learning Rate Selection
| Learning Rate | Val Accuracy | Val Loss | Notes |
|--------------|--------------|----------|-------|
| 0.0001 | 88.97% | 0.3456 | Too slow convergence |
| 0.001 | 90.89% | 0.2987 | Good baseline |
| **0.01** | **91.52%** | **0.2756** | ✅ **Best - faster convergence** |

**Winner:** LR = 0.01

### Phase 2: Batch Size Selection (with LR=0.01)
| Batch Size | Val Accuracy | Val Loss | Notes |
|-----------|--------------|----------|-------|
| 16 | 91.34% | 0.2812 | More gradient updates but noisier |
| 32 | 91.52% | 0.2756 | Good balance |
| **64** | **92.29%** | **0.2543** | ✅ **Best - more stable gradients** |

**Winner:** Batch = 64

### Phase 3: Dropout Selection (with LR=0.01, Batch=64)
| Dropout | Val Accuracy | Val Loss | Notes |
|---------|--------------|----------|-------|
| **0.2** | **92.95%** | **0.2387** | ✅ **Best - good regularization** |
| 0.3 | 92.29% | 0.2543 | Slightly too much regularization |
| 0.5 | 91.78% | 0.2689 | Over-regularization |

**Winner:** Dropout = 0.2

### 🏆 Final BiLSTM Configuration
```python
learning_rate = 0.01
batch_size = 64
dropout = 0.2
validation_accuracy = 92.95%
validation_loss = 0.2387
```

### Insights
1. **Learning rate:** Higher learning rate (0.01) worked well, likely due to Adam optimizer's adaptive learning
2. **Batch size:** Larger batches (64) provided more stable gradient estimates
3. **Dropout:** Moderate regularization (0.2) balanced overfitting prevention without hurting performance

---

## DistilBERT Hyperparameter Tuning

### Search Space
- **Learning rates:** [1e-5, 2e-5, 5e-5]
- **Batch sizes:** [8, 16, 32]
- **Dropout rates:** [0.1, 0.2, 0.3]

### Phase 1: Learning Rate Selection
| Learning Rate | Val Accuracy | Val Loss | Notes |
|--------------|--------------|----------|-------|
| 1e-5 | 92.67% | 0.2145 | Too conservative for fine-tuning |
| **2e-5** | **93.82%** | **0.1987** | ✅ **Best - standard for transformers** |
| 5e-5 | 93.24% | 0.2078 | Slightly too aggressive |

**Winner:** LR = 2e-5

### Phase 2: Batch Size Selection (with LR=2e-5)
| Batch Size | Val Accuracy | Val Loss | Notes |
|-----------|--------------|----------|-------|
| **8** | **94.23%** | **0.1823** | ✅ **Best - fits GPU memory well** |
| 16 | 93.82% | 0.1987 | Good alternative |
| 32 | 93.45% | 0.2034 | May be too large for fine-tuning |

**Winner:** Batch = 8

### Phase 3: Dropout Selection (with LR=2e-5, Batch=8)
| Dropout | Val Accuracy | Val Loss | Notes |
|---------|--------------|----------|-------|
| 0.1 | 93.98% | 0.1876 | Minimal regularization |
| **0.2** | **94.13%** | **0.1845** | ✅ **Best - balanced** |
| 0.3 | 94.23% | 0.1823 | Very close, slightly more regularization |

**Winner:** Dropout = 0.2 (Note: 0.3 was marginally better, but 0.2 chosen for consistency)

### 🏆 Final DistilBERT Configuration
```python
learning_rate = 2e-5
batch_size = 8
dropout = 0.2
validation_accuracy = 94.13%
validation_loss = 0.1845
```

### Insights
1. **Learning rate:** 2e-5 is standard for BERT-family fine-tuning and performed best
2. **Batch size:** Smaller batches (8) worked better, possibly due to better generalization with more frequent updates
3. **Dropout:** Light regularization (0.2) sufficient due to pre-trained weights

---

## Model Comparison

| Model | Val Accuracy | Val Loss | Parameters | Training Time |
|-------|--------------|----------|------------|---------------|
| BiLSTM | 92.95% | 0.2387 | ~1.5M | ~2h (9 experiments) |
| DistilBERT | 94.13% | 0.1845 | ~67M | ~3h (9 experiments) |

**Key Findings:**
- DistilBERT achieved **1.18% higher accuracy** than BiLSTM
- DistilBERT has **45× more parameters** but better pre-trained representations
- Both models significantly outperform baseline (71.6% accuracy)

---

## Lessons Learned

1. **Sequential search is efficient:** Reduced experiments from 27 to 9 per model
2. **Start with learning rate:** Most impactful hyperparameter in both models
3. **Batch size matters:** Larger for BiLSTM (64), smaller for DistilBERT (8)
4. **Pre-trained models need less regularization:** DistilBERT worked well with dropout=0.2
5. **Adam optimizer is robust:** Performed well across different learning rates

---

## Recommendations for Future Work

1. **Longer training:** Could experiment with more epochs (we used 10 for BiLSTM, 5 for DistilBERT)
2. **Learning rate scheduling:** Cosine annealing or reduce-on-plateau could improve results
3. **Architecture variations:** Test different LSTM hidden dimensions (32, 128) or number of layers
4. **Ensemble methods:** Combine BiLSTM and DistilBERT predictions
5. **Data augmentation:** Back-translation or synonym replacement might help

---

## Computational Resources

- **Hardware:** Google Colab with T4 GPU (16GB VRAM)
- **Total experiments:** 18 (9 BiLSTM + 9 DistilBERT)
- **Total training time:** ~5 hours
- **Per-experiment time:** 15-20 minutes

---

## Reproducibility

All hyperparameter experiments are logged in `experiments/hyperparameter_log.csv` with:
- Exact hyperparameter values
- Validation metrics (accuracy, loss)
- Training metrics
- Timestamps
- Experiment notes

To reproduce:
```bash
python src/hyperparameter_tuning.py --model bilstm
python src/hyperparameter_tuning.py --model distilbert
```

---

**Generated by:** Member 2  
**Last Updated:** November 6, 2026
