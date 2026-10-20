# Emotion Detector

A web app that detects the emotion in short text (**joy, sadness, anger, fear, love, surprise**) and highlights the words behind the prediction.

We use the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset to compare three approaches:

1. TF-IDF + Logistic Regression (baseline)
2. BiLSTM
3. Fine-tuned DistilBERT

Models are compared on **macro-F1, latency, and bias**. The best model will be served through an API with a Streamlit frontend.

**Track:** A (Application Development)

## Problem Statement

Understanding emotion in online text (reviews, support chats, mental-wellness apps) is hard, and plain positive/negative sentiment is not enough. This project classifies text into six emotions and explains which words drove the prediction.

## Target Audience

- Customer support teams
- Content moderators
- Mental-wellness app developers
- Researchers interested in explainable emotion detection

## Why Machine Learning?

Keyword or rule-based systems fail on negation ("I'm not happy"), sarcasm, and context. ML/DL models learn these patterns from data. Comparing a simple baseline against a transformer shows how much context-awareness helps.

## Dataset

- Source: [dair-ai/emotion on Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)
- Raw splits: 16,000 train / 2,000 validation / 2,000 test
- After cleaning: 15,594 train / 1,958 validation / 1,977 test
- Labels: 0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise
- The dataset is class-imbalanced: joy (5,233) and sadness (4,557) dominate the cleaned training set, while surprise has only 552 samples (about 9.5x fewer than joy). This is why macro-F1 and class weights are used.

Raw and processed data are **not** stored in this repo (see `.gitignore`). They are created by the scripts below.

## Project Structure

```
Emotion_Detector/
├── README.md
├── .gitignore
├── requirements.txt
├── Data_Scripts/            # Phase 2 code
│   ├── 01_download_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   └── 04_features.py
├── data/                    # created by the scripts (gitignored)
│   ├── raw/
│   └── processed/
├── Figures/                 # EDA charts used in the report
└── Reports/                 # LaTeX report (report.tex, report.pdf)
```

## Setup

```bash
git clone https://github.com/l242585-ai/Emotion_Detector.git
cd Emotion_Detector
python -m venv venv
venv\Scripts\activate        # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Usage (Phase 2 pipeline)

Run the scripts in order from the project root:

```bash
python Data_Scripts/01_download_data.py   # download dataset from Hugging Face to data/raw/
python Data_Scripts/02_clean_data.py      # missing values, duplicates, outliers, leakage check
python Data_Scripts/03_eda.py             # EDA charts saved to Figures/
python Data_Scripts/04_features.py        # TF-IDF, tokenization, label encoding, class weights
```

Each script depends on the output of the previous one (`04_features.py` needs the cleaned files created by `02_clean_data.py`). `03_eda.py` is only needed to regenerate the charts.

## Phase 2 Summary

- **Collection:** dataset downloaded with the Hugging Face `datasets` library (`01_download_data.py`).
- **Cleaning (`02_clean_data.py`):** missing value check, duplicate removal, text-length outlier removal with the IQR rule, and a train/validation/test overlap (leakage) check. Cleaned files are saved as `data/processed/*_cleaned.csv`.
- **EDA (`03_eda.py`):** class distribution across splits, text length distribution, word clouds, top words per emotion, and emotion proportions per split.
- **Features (`04_features.py`):**
  - Label encoding (labels 0-5)
  - TF-IDF (unigrams + bigrams, max 20,000 features, sublinear TF) for Logistic Regression
  - DistilBERT tokenizer (`distilbert-base-uncased`, max length 128, padding and truncation) for the deep models
  - Balanced class weights to handle imbalance

## Report

The Phase 1 and Phase 2 technical report (LaTeX) is in `Reports/`: `report.tex` and the compiled `report.pdf`.

## Project Status

- [x] Phase 1: Problem definition
- [x] Phase 2: Data acquisition and preprocessing (Deliverable 1)
- [ ] Model training and comparison
- [ ] Explainability (word highlighting)
- [ ] API and Streamlit frontend

## Team

| Name | Roll No. | Contribution |
|------|----------|--------------|
| Annushay Fatima | 24L-2585 | Repository setup, data download script, README, problem definition |
| Gul E Zara | 24L-2592 | Data cleaning and validation (`02_clean_data.py`) |
| Rumesha Naveed | 24L-2603 | EDA and feature engineering (`03_eda.py`, `04_features.py`), figures |

## Course

AI Project, Deliverable 1: Problem Definition & Data Foundations.
