# Emotion Detector

A web app that detects the emotion in short text (**joy, sadness, anger, fear, love, surprise**) and highlights the words behind the prediction.

We use the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset to compare three approaches:

1. TF-IDF + Logistic Regression (baseline)
2. BiLSTM
3. Fine-tuned DistilBERT

Models are compared on **macro-F1, latency, and bias**. The best model will be served through an API with a Streamlit frontend.

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
- Splits: 16,000 train / 2,000 validation / 2,000 test
- Labels: 0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise
- The dataset is class-imbalanced (joy and sadness are common, surprise is rare), which is why macro-F1 is used.

Raw data is **not** stored in this repo. It is downloaded by the scripts below.

## Project Structure

```
Emotion_Detector/
├── README.md
├── .gitignore
├── requirements.txt
├── data_scripts/        # Phase 2 code
│   ├── 01_download_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   └── 04_features.py
├── data/                # raw/processed data (gitignored)
├── figures/             # EDA charts used in the report
└── report/              # LaTeX report
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
python data_scripts/01_download_data.py   # download dataset from Hugging Face
python data_scripts/02_clean_data.py      # missing values, duplicates, outliers
python data_scripts/03_eda.py             # EDA charts saved to figures/
python data_scripts/04_features.py        # TF-IDF, tokenization, encoding
```

## Phase 2 Summary

- **Collection:** dataset downloaded via the Hugging Face `datasets` library.
- **Cleaning:** missing value check, duplicate removal, train/test leakage check, outlier handling on text length (IQR).
- **Features:** label encoding, TF-IDF (Logistic Regression), tokenizer and padding (BiLSTM), DistilBERT tokenizer, class weights for imbalance.
- **EDA:** class distribution, text length distribution, top words per emotion.

## Project Status

- [x] Phase 1: Problem definition
- [x] Phase 2: Data acquisition and preprocessing (Deliverable 1)
- [ ] Model training and comparison
- [ ] Explainability (word highlighting)
- [ ] API and Streamlit frontend

## Team

| Name | Role |
|------|------|
| Anoushay | _add role/contribution_ |
| _Member 2_ | _add role/contribution_ |
| _Member 3_ | _add role/contribution_ |

## Course

AI Project, Deliverable 1: Problem Definition & Data Foundations.
