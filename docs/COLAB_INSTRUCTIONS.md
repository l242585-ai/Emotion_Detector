# 🚀 Google Colab — Step-by-Step Guide
## Emotion Detector: Member 2 Training

---

## Prerequisites

Before starting, make sure you have:
- A **Google Account**
- Your `Emotion_Detector` project folder on **Google Drive**

---

## 📁 Step 1: Upload Project to Google Drive

### Option A — Manual Upload (One-time)
1. Go to [drive.google.com](https://drive.google.com)
2. Click **"+ New"** → **"Folder upload"**
3. Select your `Emotion_Detector` folder from your Desktop
4. Wait for upload to complete (~5–10 minutes depending on data size)

### Option B — Google Drive Desktop App (Recommended — Auto-sync)
1. Download [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Install and sign in with your Google account
3. Your Drive will appear as a folder on your PC (e.g., `G:\My Drive`)
4. Copy `Emotion_Detector` into it → it syncs automatically
5. Any results saved in Colab will appear on your PC instantly ✅

---

## 🌐 Step 2: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **"File"** → **"Upload notebook"**
3. Select `Emotion_Detector_Training.ipynb` from your project folder
4. The notebook will open in Colab

---

## ⚡ Step 3: Enable GPU Runtime (CRITICAL)

> DistilBERT is **10-20x faster** on GPU. Do this BEFORE running any code.

1. Click **"Runtime"** (top menu)
2. Click **"Change runtime type"**
3. Set **Hardware accelerator** to **"T4 GPU"**
4. Click **"Save"**

You'll see a green GPU indicator in the top-right corner ✅

---

## ▶️ Step 4: Run the Notebook

The notebook has 12 steps — run them **in order**:

| Step | Task | Time (GPU) |
|------|------|------------|
| 1 | Mount Google Drive | ~30 sec |
| 2 | Install dependencies | ~2 min |
| 3 | Verify environment | ~10 sec |
| 4 | Train Baseline | ~1 min |
| 5 | Tune BiLSTM HPs | ~15 min |
| 6 | Train Final BiLSTM | ~5 min |
| 7 | Tune DistilBERT HPs | ~25 min |
| 8 | Train Final DistilBERT | ~10 min |
| 9 | Evaluate all models | ~3 min |
| 10 | Generate visualizations | ~1 min |
| 11 | View HP summary table | ~5 sec |
| 12 | Deliverables checklist | ~5 sec |

**Total estimated time: ~60-70 minutes on T4 GPU**

To run all at once: **Runtime → Run all**

---

## ⚠️ Important Notes

### Session Disconnection
- Colab disconnects after **~90 minutes of idle** (free tier)
- If disconnected, **your Drive files are safe** (they're already saved to Drive)
- Just reconnect and run from where you left off

### RAM Limits
- Free Colab: ~12 GB RAM
- If you get "RAM exceeded" during DistilBERT:
  - Reduce `batch_size` from 16 to 8 in Step 7
  - Or use `batch_size=8` directly in the tuning config

### Colab Pro (Optional)
- If you have Colab Pro, use **A100 GPU** for 3x faster training
- Select: Runtime → Change runtime type → A100

---

## 📥 Step 5: Download Results

After training, your results are already in **Google Drive** at:
```
My Drive/Emotion_Detector/
  ├── models/
  │     ├── best_model.pth        ← best overall model
  │     ├── best_bilstm.pth       ← BiLSTM weights
  │     └── best_distilbert.pth   ← DistilBERT weights
  ├── results/
  │     ├── predictions.csv       ← all predictions
  │     ├── metrics_summary.json  ← all metrics
  │     ├── training_curves.png
  │     ├── model_comparison.png
  │     ├── confusion_matrices.png
  │     └── per_class_f1.png
  └── experiments/
        └── hyperparameter_log.csv
```

If using **Google Drive Desktop**, these files sync to your PC automatically.
Otherwise, right-click any file in Drive and **Download**.

---

## 🆘 Troubleshooting

### "Module not found" errors
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/Emotion_Detector')
```
Add this at the top of any cell that fails.

### "CUDA out of memory"
Reduce batch size:
```python
# In Step 7 (DistilBERT tuning), change:
'batch_values': [8, 16, 32]  →  'batch_values': [4, 8, 16]
```

### Drive not mounted / path error
Make sure Step 1 ran successfully and `PROJECT_PATH` is correct:
```python
PROJECT_PATH = '/content/drive/MyDrive/Emotion_Detector'
```
Check your actual Drive path: the folder name must match exactly.

### Colab disconnects during long training
- Enable "Stay connected" using a browser extension
- Or break tuning into separate runs (each step saves progress)

---

## 🖥️ Alternative: Run Locally in VS Code (Without DistilBERT)

If you prefer VS Code:

```bash
# Navigate to project
cd C:\Users\Mubasher Ashfaq\Desktop\Emotion_Detector

# Activate virtual environment
.\venv\Scripts\activate

# Quick test (verify setup)
python run_member2.py --test-only

# Train baseline only
python src/train_baseline.py

# Train BiLSTM with tuning
python src/hyperparameter_tuning.py --model bilstm
python src/train_deep.py

# Skip DistilBERT (CPU-only full pipeline)
python run_member2.py --skip-distilbert

# Evaluate and plot
python src/evaluate_all.py
python src/visualize_results.py
```

> ⚠️ DistilBERT on CPU = ~3-4 hours per training run. **Not recommended locally.**

---

*Generated for Emotion Detector Project — Member 2 Pipeline*
