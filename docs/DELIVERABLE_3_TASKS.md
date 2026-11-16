# Deliverable 3 — Task Division

**Deadline:** 17 November 2026  
**Track:** A — Application Development

---

## 📋 Requirements summary

| Phase | What is needed |
|-------|----------------|
| Phase 5 (Track A) | Frontend (Streamlit) + FastAPI backend for real-time inference |
| Phase 6 | Final LaTeX report: Abstract, Visualisations, all phases merged |
| Submission | Production-ready GitHub repo, `requirements.txt`, `README.md`, LaTeX PDF |

---

## 👤 Member 1 — Annushay Fatima (24L-2585) · Backend API & Model Serving ✅

### What she built

| File | Description |
|------|-------------|
| `backend/app.py` | FastAPI server with CORS, structured logging, OpenAPI docs |
| `backend/model_service.py` | Loads all three models, unified `predict()` interface |
| `backend/explainer.py` | LIME-style leave-one-out word importance scorer |
| `backend/requirements.txt` | Backend-only pinned dependencies |
| `docs/DELIVERABLE_3_TASKS.md` | This file |
| `requirements.txt` | Root requirements updated with FastAPI + uvicorn |
| `README.md` | Updated with Deliverable 3 section: backend setup, API usage, run instructions |

### Endpoints delivered

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Welcome + health summary |
| `GET` | `/health` | Detailed health check |
| `GET` | `/models` | List all models + load status |
| `POST` | `/predict` | Return emotion, confidence, all probabilities |
| `POST` | `/predict/explain` | Prediction + per-word colour-coded highlights |

### How to run the backend

```bash
# From the project root (venv activated)
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs → http://localhost:8000/docs

### Sample curl request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so happy today!", "model": "distilbert"}'
```

---

## 👤 Member 2 — Gul E Zara (24L-2592) · Frontend (Streamlit)

### What she will build

| File | Description |
|------|-------------|
| `frontend/app.py` | Streamlit UI |
| `frontend/requirements.txt` | Frontend-only dependencies |

### Features

- Text input box → calls `POST /predict` → shows emotion + emoji
- Confidence bar chart for all 6 emotions (Plotly / Altair)
- Highlighted-word display (uses `GET /predict/explain` highlights)
- Model selector dropdown (Baseline / BiLSTM / DistilBERT)
- Sample text buttons for quick testing

### How she will run it

```bash
# Backend must be running on port 8000 first (see Member 1)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Member 3 — Rumesha Naveed (24L-2603) · Final Report & Visualisations

### What she will build

| File | Description |
|------|-------------|
| `docs/report/deliverable3.tex` | Final LaTeX report (all phases merged) |
| `docs/Reports/Deliverable_3_Report.pdf` | Compiled PDF |
| `figures/` | Updated high-quality plots |

### Sections required

- Abstract
- Introduction and problem statement
- Dataset and preprocessing (Phase 1 & 2)
- Model architectures (Phase 3)
- Hyperparameter tuning and training (Phase 3 & 4)
- Evaluation and comparative analysis
- Deployment architecture (API + Streamlit)
- Results discussion and limitations
- References

---

## ✅ Status

| Member | Status |
|--------|--------|
| Annushay (Backend) | ✅ Complete |
| Gul E Zara (Frontend) | 🔲 To do |
| Rumesha (Report) | 🔲 To do |
