# Emotion Detector

A web application that detects the emotion in short English text (**joy, sadness, anger, fear, love, surprise**) and highlights the words behind the prediction.

We compare three approaches on the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset:

| Model | Test Accuracy | Macro-F1 |
|-------|--------------|----------|
| TF-IDF + Logistic Regression (Baseline) | 86.5 % | ~0.82 |
| BiLSTM | 91.1 % | 0.872 |
| DistilBERT (fine-tuned) | **93.0 %** | **0.880** |

**Track:** A (Application Development)

---

## Problem Statement

Understanding emotion in online text (reviews, support chats, mental-wellness apps) requires more than positive/negative sentiment. This project classifies text into six fine-grained emotions and explains which words drove the prediction.

## Dataset

- Source: [dair-ai/emotion on Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)
- Splits: 16 000 train / 2 000 validation / 2 000 test (after cleaning: 15 594 / 1 958 / 1 977)
- Labels: 0 sadness · 1 joy · 2 love · 3 anger · 4 fear · 5 surprise
- Class-imbalanced: joy (5 233) and sadness (4 557) dominate; surprise has only 552 samples. Macro-F1 and class weights compensate.

---

## Project Structure

```
Emotion_Detector/
├── README.md
├── requirements.txt          ← root install (training + API + frontend)
├── .gitignore
│
├── backend/                  ← FastAPI REST API (Member 1)
│   ├── app.py                ← server: routes, CORS, OpenAPI docs
│   ├── model_service.py      ← load models + predict()
│   ├── explainer.py          ← LIME-style word importance
│   └── requirements.txt      ← backend-only dependencies
│
├── frontend/                 ← Streamlit UI (Member 2)
│   ├── app.py
│   └── requirements.txt
│
├── src/                      ← training source code
│   ├── models/
│   │   ├── baseline_model.py
│   │   ├── deep_model.py
│   │   └── distilbert_model.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   └── metrics.py
│   ├── train_baseline.py
│   ├── train_deep.py
│   ├── train_distilbert.py
│   ├── evaluate.py
│   ├── evaluate_all.py
│   ├── error_analysis.py
│   ├── hyperparameter_tuning.py
│   └── visualize_results.py
│
├── scripts/data_processing/  ← data pipeline (run in order)
│   ├── 01_download_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   └── 04_features.py
│
├── notebooks/                ← Google Colab training notebook
├── models/                   ← saved model files (gitignored)
│   ├── best_baseline.pkl
│   ├── best_bilstm.pth
│   └── best_model.pth        ← DistilBERT
├── results/                  ← JSON metrics + plots
├── figures/                  ← EDA + architecture diagrams
├── experiments/              ← hyperparameter logs
└── docs/
    ├── DELIVERABLE_3_TASKS.md
    ├── ARCHITECTURE.md
    ├── report/               ← LaTeX source
    └── Reports/              ← compiled PDFs
```

---

## Setup

```bash
git clone https://github.com/l242585-ai/Emotion_Detector.git
cd Emotion_Detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

---

## Reproducing the pipeline (Phases 1 – 4)

Run from the project root in this order:

```bash
python scripts/data_processing/01_download_data.py   # download from Hugging Face
python scripts/data_processing/02_clean_data.py      # clean, deduplicate
python scripts/data_processing/03_eda.py             # EDA figures → figures/
python scripts/data_processing/04_features.py        # TF-IDF + tokenisation + class weights

python src/train_baseline.py                         # TF-IDF + LogReg → models/best_baseline.pkl
python src/train_deep.py                             # BiLSTM → models/best_bilstm.pth
python src/train_distilbert.py                       # DistilBERT → models/best_model.pth

python src/evaluate_all.py                           # evaluate all three models
python src/error_analysis.py                         # error analysis
python src/visualize_results.py                      # generate result plots
```

---

## Deliverable 3 — Running the Application

### 1 · Start the backend API

```bash
# From the project root (venv activated, models must be trained)
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The API starts at **http://localhost:8000**  
Interactive Swagger docs: **http://localhost:8000/docs**

#### Quick test with curl

```bash
# Simple prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel absolutely wonderful today!", "model": "distilbert"}'

# Prediction with word highlights
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"text": "I am terrified of what might happen.", "model": "distilbert"}'

# Health check
curl http://localhost:8000/health
```

#### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Welcome message + available models |
| `GET` | `/health` | Health status + loaded model names |
| `GET` | `/models` | All models with descriptions and load status |
| `POST` | `/predict` | Emotion + confidence + all 6 probabilities |
| `POST` | `/predict/explain` | Above + per-word colour-coded highlights |

#### `POST /predict` request format

```json
{
  "text": "I feel so happy today!",
  "model": "distilbert"
}
```

Omit `"model"` to let the API pick the best loaded model automatically.

#### `POST /predict` response format

```json
{
  "emotion": "joy",
  "emotion_id": 1,
  "emoji": "😊",
  "confidence": 0.9731,
  "probabilities": {
    "sadness": 0.0043,
    "joy":     0.9731,
    "love":    0.0135,
    "anger":   0.0032,
    "fear":    0.0027,
    "surprise":0.0032
  },
  "model": "distilbert",
  "latency_ms": 48.3
}
```

### 2 · Start the Streamlit frontend

```bash
# In a separate terminal (backend must be running on port 8000)
cd frontend
streamlit run app.py
```

The UI opens at **http://localhost:8501**

---

## Deliverable 3 — Architecture

```
User (browser)
     │
     ▼
Streamlit frontend  (port 8501)
  frontend/app.py
     │  POST /predict / /predict/explain
     ▼
FastAPI backend     (port 8000)
  backend/app.py
  backend/model_service.py  ← loads pkl / .pth files from models/
  backend/explainer.py      ← LIME-style word importance
     │
     ▼
Trained models  (models/ directory)
  best_baseline.pkl   TF-IDF + LogReg
  best_bilstm.pth     BiLSTM
  best_model.pth      DistilBERT
```

**Yes — the frontend and backend are separate processes.** The Streamlit app calls the FastAPI server over HTTP. This separation means you can swap either side independently and test the API without the UI.

---

## Team

| Name | Roll No. | Deliverable 3 Contribution |
|------|----------|-----------------------------|
| Annushay Fatima | 24L-2585 | FastAPI backend, model service, LIME explainer, requirements, README |
| Gul E Zara | 24L-2592 | Streamlit frontend |
| Rumesha Naveed | 24L-2603 | Final LaTeX report + visualisations |

## Course

AI Project — Deliverable 3: Final Deployment & Technical Report.
