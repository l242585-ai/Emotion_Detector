# Task Division - Deliverable 1 (Phase 1 & 2)

**Deadline:** March 18, 2024  
**Project:** Emotion Detection from Text  
**Track:** Application Development (Track A)

---

## Team Members & Responsibilities

### 👤 Member 1: [Name] - Data Collection & Setup
**Primary Tasks:**
- ✅ Download data from Hugging Face (`01_download_data.py`)
- ✅ Create project folder structure
- ✅ Set up `.gitignore` and requirements.txt
- Update README.md with setup instructions
- Document data source and statistics
- **Report Section:** Introduction, Problem Statement, Dataset Description

**Files to Own:**
- `data_scripts/01_download_data.py`
- `README.md`
- `.gitignore`
- LaTeX: Introduction + Dataset sections

**Time Estimate:** 3-4 hours

---

### 👤 Member 2: [Name] - Data Cleaning & Validation
**Primary Tasks:**
- Write `02_clean_data.py`:
  - Check for missing values
  - Remove duplicates
  - Handle text length outliers (IQR method)
  - Verify no train/test leakage
  - Save cleaned data
- Create data validation report (stats before/after cleaning)
- **Report Section:** Data Cleaning Methodology, Results

**Files to Own:**
- `data_scripts/02_clean_data.py`
- LaTeX: Data Cleaning section

**Time Estimate:** 4-5 hours

---

### 👤 Member 3: [Name] - EDA & Feature Engineering
**Primary Tasks:**
- Write `03_eda.py`:
  - Class distribution bar chart
  - Text length distribution histogram
  - Word clouds per emotion
  - Top words per emotion
  - Correlation analysis
- Write `04_features.py`:
  - TF-IDF vectorization setup
  - Tokenization for deep learning
  - Label encoding
  - Calculate class weights for imbalance
- **Report Section:** Exploratory Data Analysis, Feature Engineering

**Files to Own:**
- `data_scripts/03_eda.py`
- `data_scripts/04_features.py`
- `figures/` directory with all charts
- LaTeX: EDA + Feature Engineering sections

**Time Estimate:** 5-6 hours

---

## Shared Responsibilities (Everyone)

1. **Code Review:** Review each other's pull requests
2. **Testing:** Run all scripts end-to-end before submission
3. **LaTeX Report:** 
   - Each member writes their section
   - One person compiles and formats
   - Everyone reviews final PDF
4. **Git Commits:** Make regular, meaningful commits (not all at once!)

---

## Timeline (Recommended)

| Date | Milestone | Who |
|------|-----------|-----|
| **Day 1-2** | Data collection done, folder structure set | Member 1 |
| **Day 2-3** | Cleaning script complete | Member 2 |
| **Day 3-5** | EDA visualizations ready | Member 3 |
| **Day 5-6** | Feature engineering complete | Member 3 |
| **Day 6-7** | LaTeX report first draft | Everyone |
| **Day 7-8** | Code review, testing, integration | Everyone |
| **Day 8-9** | Final report compilation | Everyone |
| **Day 9** | Buffer day for issues | Everyone |
| **Day 10** | Submit 24h before deadline | Everyone |

---

## Communication Protocol

- **Daily Standup:** Quick message in group chat
- **Code Questions:** Use GitHub Issues or pull request comments
- **Meetings:** At least 2 sync meetings (start + mid-point)
- **Emergency:** If stuck >2 hours, ask for help immediately

---

## Deliverable Checklist

### GitHub Repository
- [ ] Professional README.md
- [ ] `.gitignore` (exclude data/ folder)
- [ ] `requirements.txt` with all dependencies
- [ ] `/data_scripts/` with all 4 Python scripts
- [ ] Commit history shows all 3 members actively contributing
- [ ] Repository is public or shared with instructor

### Technical Report (LaTeX)
- [ ] Problem statement and justification
- [ ] Target audience identification
- [ ] AI/ML necessity explanation
- [ ] Dataset description with statistics
- [ ] Data cleaning methodology
- [ ] EDA with visualizations (≥ 4 charts)
- [ ] Feature engineering strategy
- [ ] References in BibTeX format

### Scripts Must Run Without Errors
```bash
python data_scripts/01_download_data.py
python data_scripts/02_clean_data.py
python data_scripts/03_eda.py
python data_scripts/04_features.py
```

---

## Git Workflow (Important!)

1. **Each member works on their branch:**
   ```bash
   git checkout -b feature/member1-data-download
   git checkout -b feature/member2-data-cleaning
   git checkout -b feature/member3-eda-features
   ```

2. **Make small, regular commits:**
   ```bash
   git add data_scripts/01_download_data.py
   git commit -m "Add data download script with HuggingFace integration"
   git push origin feature/member1-data-download
   ```

3. **Create Pull Requests for review**

4. **Merge to main after approval**

This ensures the commit history shows everyone's contribution!

---

## Notes

- **Data Size:** Do NOT upload raw data to GitHub (it's in `.gitignore`)
- **Python Version:** Use Python 3.8+ for compatibility
- **Dependencies:** Test that `requirements.txt` has everything needed
- **LaTeX Template:** Use a clean academic template (IEEE, ACM, or simple article)

---

## Questions?

If stuck, reach out to:
1. Group members first (< 30 min response time)
2. Check project instructions document
3. Email TA/instructor for clarifications

**Remember:** Submit 24 hours BEFORE the deadline (March 17, 11:59 PM)
