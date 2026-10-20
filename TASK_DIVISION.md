# Task Division - Deliverable 1 (Phase 1 & 2)

**Project:** Emotion Detection from Text with Explainable Predictions
**Track:** Application Development (Track A)

---

## Team Members & Responsibilities

### Member 1: Annushay Fatima (24L-2585) - Data Collection & Setup
**Tasks:**
- [x] Download data from Hugging Face (`01_download_data.py`)
- [x] Create project folder structure
- [x] Set up `.gitignore` and `requirements.txt`
- [x] Write and update `README.md`
- [x] Report sections: Phase 1 (Problem Statement, Target Audience, AI Justification) and Data Collection

**Files owned:**
- `Data_Scripts/01_download_data.py`
- `README.md`, `.gitignore`
- LaTeX: Phase 1 + Data Collection sections

---

### Member 2: Gul E Zara (24L-2592) - Data Cleaning & Validation
**Tasks:**
- [x] Write `02_clean_data.py`:
  - Check for missing values
  - Remove duplicates
  - Handle text length outliers (IQR method)
  - Check train/validation/test leakage
  - Save cleaned data to `data/processed/`
- [x] Run download and cleaning scripts
- [x] Copy cleaning statistics (missing, duplicates, outliers, leakage overlap) into the report table
- [x] Report section: Data Cleaning

**Files owned:**
- `Data_Scripts/02_clean_data.py`
- LaTeX: Data Cleaning section

---

### Member 3: Rumesha Naveed (24L-2603) - EDA & Feature Engineering
**Tasks:**
- [x] Write `03_eda.py` (class distribution, text length histogram, word clouds, top words per emotion, emotion proportions per split)
- [x] Write `04_features.py` (TF-IDF, DistilBERT tokenization, label encoding, class weights)
- [x] Save all charts in `Figures/`
- [x] Add a true feature-correlation chart (word count / character count vs emotion)
- [x] Run `04_features.py` end to end and copy the TF-IDF vocabulary size into the report
- [x] Report sections: Exploratory Data Analysis, Feature Strategy

**Files owned:**
- `Data_Scripts/03_eda.py`
- `Data_Scripts/04_features.py`
- `Figures/` directory with all charts
- LaTeX: EDA + Feature Strategy sections

---

## Shared Responsibilities (Everyone)

1. **Code review:** review each other's pull requests.
2. **Testing:** run all scripts end to end before submission.
3. **LaTeX report:** each member checks their section; one person compiles; everyone reviews the final PDF.
4. **Git commits:** regular, meaningful commits from every member (not all at once).

---

## Status

| Item | Status |
|------|--------|
| Data download, folder structure, README | Done |
| Cleaning script and statistics in report | Done |
| EDA charts and correlation chart | Done |
| Feature engineering script (final run, vocabulary size noted) | Done |
| LaTeX report (Phase 1 + Phase 2) | Done |
| Final PDF in `Reports/` and merge to `main` | Done |


---

## Deliverable Checklist

### GitHub Repository
- [x] Professional `README.md`
- [x] `.gitignore` (excludes `data/`)
- [x] `requirements.txt`
- [x] `Data_Scripts/` with all 4 Python scripts
- [x] Commit history shows all 3 members contributing
- [x] Repository shared with instructor

### Technical Report (LaTeX)
- [x] Problem statement and justification
- [x] Target audience
- [x] AI/ML necessity explanation
- [x] Dataset description with statistics
- [x] Data cleaning methodology
- [x] EDA with visualizations
- [x] Feature engineering strategy
- [x] All values filled from the script outputs
- [x] Final PDF compiled; `AI_phase1.tex` + `AI_phase1.pdf` copied to `Reports/`

### Scripts Must Run Without Errors
```bash
python Data_Scripts/01_download_data.py
python Data_Scripts/02_clean_data.py
python Data_Scripts/03_eda.py
python Data_Scripts/04_features.py
```

---

## Git Workflow

1. Each member works on their own branch:
```bash
   git checkout -b <name>/<task>      # e.g. anoushay/data-scripts
```
2. Make small, regular commits and push:
```bash
   git add <files>
   git commit -m "Short, clear message"
   git push -u origin <branch>
```
3. Open a pull request on GitHub and merge to `main` after a quick review.
4. Before starting new work: `git checkout main` then `git pull origin main`.

---

## Notes

- **Data size:** do not upload raw data to GitHub (`data/` is in `.gitignore`).
- **Python version:** 3.8+.
- **Dependencies:** install with `pip install -r requirements.txt`.
- **Folder names:** `Data_Scripts`, `Figures`, `Reports` (same capitalisation in all commands and in the README).
