# Deliverable 2: Task Distribution
## Model Architecture & Empirical Evaluation

**Deadline:** 10th November, 2026

---

## Sequential Pipeline: Member 1 → Member 2 → Member 3

Each member completes their phase and hands off to the next person.

---

### 👤 Member 1: Model Architecture & Implementation (Phase 3)

**Timeframe:** Days 1-3

#### Tasks:
1. **Design Model Architecture**
   - Choose framework (PyTorch/TensorFlow/Scikit-Learn)
   - Design neural network layers (LSTM/GRU/CNN/Transformer)
   - Define input/output dimensions for emotion classification
   - Document architecture design decisions

2. **Implement Model Class**
   - Write model architecture code
   - Define loss function (e.g., CrossEntropyLoss)
   - Configure optimizer (e.g., Adam, SGD)
   - Add regularization (dropout, batch norm)

3. **Setup Training Infrastructure**
   - Create data loaders for train/val/test splits
   - Implement basic training loop
   - Add checkpoint saving functionality
   - Implement logging for metrics

4. **Create Architecture Documentation**
   - Draw architecture diagram
   - Document layer dimensions
   - Explain design choices
   - Write model architecture section for report

#### Deliverables to Pass to Member 2:
- ✅ `src/models/model.py` - Complete model architecture
- ✅ `src/utils/data_loader.py` - Data loading utilities
- ✅ `src/train.py` - Training script template
- ✅ Architecture diagram for report
- ✅ Model architecture section in report

**Handoff:** Once model architecture is implemented and can run forward pass, pass to Member 2

---

### 👤 Member 2: Training & Hyperparameter Tuning (Phase 3 & 4)

**Timeframe:** Days 4-7 (starts after Member 1 completes)

#### Tasks:
1. **Complete Training Pipeline**
   - Enhance training loop with validation
   - Add early stopping
   - Implement progress tracking (tqdm/logging)
   - Add model checkpointing for best model

2. **Hyperparameter Tuning**
   - Test learning rates: [0.001, 0.0001, 0.00001]
   - Test batch sizes: [16, 32, 64]
   - Test dropout rates: [0.2, 0.3, 0.5]
   - Test number of layers/hidden units
   - Document ALL experiments in a table

3. **Train Final Model**
   - Select best hyperparameters
   - Train final model on full training data
   - Save best model weights
   - Generate predictions on test set

4. **Create Visualizations**
   - Plot training/validation loss curves
   - Plot training/validation accuracy curves
   - Save training history
   - Document hyperparameter selection justification

#### Deliverables to Pass to Member 3:
- ✅ `models/best_model.pth` - Trained model weights (link in README)
- ✅ `results/predictions.csv` - Model predictions on test set
- ✅ `experiments/hyperparameter_log.csv` - All experiments
- ✅ `results/training_curves.png` - Loss/accuracy plots
- ✅ Hyperparameter tuning section in report
- ✅ Training methodology section in report

**Handoff:** Once model is trained and predictions are saved, pass to Member 3

---

### 👤 Member 3: Evaluation & Error Analysis (Phase 4)

**Timeframe:** Days 8-10 (starts after Member 2 completes)

#### Tasks:
1. **Implement Evaluation Metrics**
   - Precision (macro, micro, weighted)
   - Recall (macro, micro, weighted)
   - F1-Score (macro, micro, weighted)
   - AUC-ROC (one-vs-rest for multi-class)
   - Confusion matrix

2. **Comprehensive Evaluation**
   - Load test set and predictions from Member 2
   - Calculate all metrics
   - Generate classification report
   - Create confusion matrix visualization
   - Plot ROC curves (if applicable)

3. **Metric Justification**
   - Explain why each metric is appropriate
   - Discuss class imbalance in dataset
   - Justify evaluation strategy
   - Compare with baseline/random classifier

4. **Error Analysis & Bias Check**
   - Identify misclassified examples
   - Analyze which emotion classes fail most
   - Check performance by text length
   - Check for bias patterns
   - Identify emotion confusion patterns (which emotions are confused?)
   - Investigate model weaknesses

5. **Final Documentation**
   - Compile all results
   - Write evaluation section
   - Write error analysis section
   - Write bias check section
   - Create contribution table for report

#### Deliverables (Final):
- ✅ `src/evaluate.py` - Evaluation script
- ✅ `src/utils/metrics.py` - Metrics implementation
- ✅ `src/error_analysis.py` - Error analysis script
- ✅ `results/evaluation_metrics.json` - All metrics
- ✅ `results/confusion_matrix.png` - Confusion matrix
- ✅ `results/classification_report.txt` - Detailed report
- ✅ `results/error_analysis.txt` - Error patterns
- ✅ Evaluation section in report
- ✅ Error analysis section in report
- ✅ Bias check section in report
- ✅ Contribution table in report

**Final Step:** Complete the report and prepare submission

---

---

## Workflow Summary

```
Member 1 (Days 1-3)
   └─> Builds Model Architecture
         └─> Hands off to Member 2

Member 2 (Days 4-7)
   └─> Trains Model & Tunes Hyperparameters
         └─> Hands off to Member 3

Member 3 (Days 8-10)
   └─> Evaluates & Analyzes Errors
         └─> Completes Report & Submission
```

---

## Handoff Checklist

### ✅ Member 1 → Member 2 Handoff:
- [ ] Model architecture code is complete and runnable
- [ ] Data loaders work correctly
- [ ] Basic training script template exists
- [ ] Architecture diagram created
- [ ] Architecture section written in report
- [ ] Code committed to Git with clear commit messages

### ✅ Member 2 → Member 3 Handoff:
- [ ] Model is fully trained
- [ ] Best model weights saved (with link in README)
- [ ] Test set predictions saved in CSV format
- [ ] Hyperparameter experiments documented
- [ ] Training curves generated
- [ ] Hyperparameter section written in report
- [ ] Code committed to Git with clear commit messages

### ✅ Member 3 → Final Submission:
- [ ] All metrics calculated
- [ ] Confusion matrix and visualizations created
- [ ] Error analysis complete
- [ ] Bias check complete
- [ ] All report sections written
- [ ] Contribution table created
- [ ] Code committed to Git with clear commit messages
- [ ] README updated with instructions

---

## Timeline (Sequential)

### Days 1-3: Member 1 Works
- Day 1: Design architecture, choose framework
- Day 2: Implement model class and data loaders
- Day 3: Complete training template, create diagrams, write report section
- **Commit frequently with clear messages**

### Days 4-7: Member 2 Works (after receiving Member 1's code)
- Day 4: Set up training pipeline, first training run
- Day 5-6: Hyperparameter tuning experiments
- Day 7: Train final model, generate predictions & visualizations
- **Commit frequently with clear messages**

### Days 8-10: Member 3 Works (after receiving Member 2's predictions)
- Day 8: Implement metrics, run evaluation
- Day 9: Error analysis and bias check
- Day 10: Complete report, create contribution table, final submission
- **Commit frequently with clear messages**

---

## File Structure

```
Emotion_Detector/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py                 # Member 1
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Member 1
│   │   ├── metrics.py               # Member 3
│   │   └── visualization.py         # Member 2
│   ├── train.py                     # Member 1 (template) + Member 2 (complete)
│   ├── evaluate.py                  # Member 3
│   └── error_analysis.py            # Member 3
├── experiments/
│   └── hyperparameter_log.csv       # Member 2
├── results/
│   ├── predictions.csv              # Member 2
│   ├── evaluation_metrics.json      # Member 3
│   ├── confusion_matrix.png         # Member 3
│   ├── classification_report.txt    # Member 3
│   ├── error_analysis.txt           # Member 3
│   └── training_curves.png          # Member 2
├── models/
│   └── best_model.pth              # Member 2 (link in README)
├── report/
│   └── main.tex                    # All members (respective sections)
└── requirements.txt                # All members (keep updated)
```

---

## Git Commit Guidelines

### Good Commit Messages:
✅ `feat(baseline): implement logistic regression baseline model`  
✅ `feat(deep): add LSTM architecture with bidirectional layers`  
✅ `feat(metrics): implement multi-class AUC-ROC calculation`  
✅ `exp(hyperparam): test learning rates [0.001, 0.0001] - best: 0.0001`  
✅ `eval(analysis): identify confusion between sadness and fear classes`  
✅ `fix(data): correct validation split stratification`  
✅ `docs(report): add hyperparameter justification section`

### Bad Commit Messages:
❌ `update`  
❌ `changes`  
❌ `fixed stuff`  
❌ `asdfgh`

### Commit Prefixes to Use:
- `feat`: New feature implementation
- `exp`: Experiment or hyperparameter tuning
- `eval`: Evaluation or analysis results
- `fix`: Bug fix
- `docs`: Documentation update
- `refactor`: Code refactoring
- `test`: Adding tests

---

## Communication & Coordination

### Simple API Contracts (No Blocking):

**Member 1's Data Loader API (for Member 2 to use):**
```python
# Simple interface Member 2 can expect
from src.utils.data_loader import load_emotion_data

train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_emotion_data()
# Returns: texts as list of strings, labels as numpy arrays
```

**Member 1 & 2's Model Predictions API (for Member 3 to use):**
```python
# What Member 3 needs from both models
predictions = model.predict(test_texts)  # Returns numpy array of predictions
# Member 3 can build this interface for testing, then both models implement it
```

### Slack/Communication Guidelines:
- **Quick Updates:** Share when you complete a major module
- **Blockers:** Immediately communicate if you hit a dependency issue
- **Integration:** Coordinate brief 15-min syncs before integration points
- **No Micromanagement:** Trust each member to own their area

---

## Important Notes

1. **Work Independently First:** Each member should be able to work for Week 1 without needing others' code
2. **Simple Interfaces:** Keep APIs between modules simple (just pass numpy arrays/lists)
3. **Test with Dummy Data:** If you need another member's output, create dummy data to test your code
4. **Individual Contributions Must Be Visible:** Each member's commits should clearly show their work on their assigned modules
5. **No Manual Uploads:** Always use Git CLI (`git add`, `git commit`, `git push`)
6. **Frequent Commits:** Commit after completing each logical unit of work
7. **Model Weights:** Upload large model files to Google Drive/Hugging Face, add link to README
8. **Requirements.txt:** Update whenever new packages are installed

---

## Success Criteria

- ✅ All three members can work in parallel for Week 1 without blocking each other
- ✅ Each member has clear ownership and Git history shows individual contributions
- ✅ Both baseline and deep learning models successfully train
- ✅ Comprehensive evaluation with all required metrics for both models
- ✅ Hyperparameter table with justifications
- ✅ Error analysis and bias check completed
- ✅ Professional Git practices followed throughout
- ✅ Complete LaTeX report with all sections
- ✅ Reproducible environment with requirements.txt

---

## Bonus: How to Work When You Need Integration

### Member 2 (Before Member 1's data loader is ready):
```python
# Week 1: Use simple pandas loading
import pandas as pd
train_df = pd.read_csv('data/processed/train_cleaned.csv')
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].values

# Week 2: Switch to Member 1's loader
from src.utils.data_loader import load_emotion_data
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_emotion_data()
```

### Member 3 (Before models are ready):
```python
# Week 1: Test metrics with dummy predictions
import numpy as np
dummy_predictions = np.random.randint(0, 6, size=100)
dummy_true_labels = np.random.randint(0, 6, size=100)
# Test all your metrics functions with this dummy data

# Week 2+: Use real model predictions
from src.models.baseline_model import BaselineModel
from src.models.deep_model import DeepModel
# Evaluate real models
```

---

**Remember:** Your Git commit history will be evaluated to assess individual contributions. Make sure your commits clearly demonstrate your work on your assigned modules!
