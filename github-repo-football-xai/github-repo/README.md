# ⚽ Football Match Result Prediction using Explainable Machine Learning

Analysis of feature importance in football match outcome prediction using Random Forest and SHAP (SHapley Additive exPlanations).

> 📄 This repository contains the source code, dataset reference, and analysis notebook accompanying the research paper *"Analysis of Feature Importance in Football Match Result Prediction Using Explainable Machine Learning."*

---

## 📌 Project Overview

This project predicts football match outcomes (**Home Win / Draw / Away Win**) using 18 team tactical attributes (e.g. defensive pressure, build-up passing, chance creation) extracted from the **European Soccer Database**. A Random Forest classifier is trained with balanced class weighting, and **SHAP TreeExplainer** is used to explain *why* the model makes each prediction — both globally and for individual matches.

| | |
|---|---|
| **Dataset** | [European Soccer Database (Kaggle)](https://www.kaggle.com/datasets/hugomathien/soccer) |
| **Matches analyzed** | 19,355 (11 European leagues, 2008–2016) |
| **Model** | Random Forest (`n_estimators=200`, `max_depth=10`, `class_weight='balanced'`) |
| **Explainability** | SHAP (TreeExplainer) — Bar, Beeswarm, Waterfall plots |
| **Accuracy** | 43.09% |
| **Weighted F1-Score** | 0.43 |

---

## 📁 Repository Structure

```
football-feature-importance-xai/
├── notebooks/
│   └── football_prediction_shap.ipynb   # Main analysis notebook
├── data/
│   └── README.md                        # Dataset download instructions
├── images/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_bar_plot.png
│   ├── shap_beeswarm_plot.png
│   └── shap_waterfall_plot.png
├── docs/
│   └── paper_english.pdf                # Full research paper
├── app.py                                # Streamlit demo app
├── model.pkl                             # Trained model (generated after running notebook)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/football-feature-importance-xai.git
cd football-feature-importance-xai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `database.sqlite` from [Kaggle - European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) and place it inside the `data/` folder.

### 4. Run the notebook
Open `notebooks/football_prediction_shap.ipynb` in Jupyter or Google Colab and run all cells. This will:
- Extract and clean match + team attribute data
- Train the Random Forest classifier
- Generate evaluation metrics and SHAP explanations
- Export the trained model as `model.pkl`

### 5. Run the Streamlit demo
```bash
streamlit run app.py
```

---

## 🧠 Methodology Summary

1. **Data Extraction** — SQL subquery joins each match with the most recent team attributes available *before* the match date (prevents data leakage and record duplication).
2. **Feature Set** — 18 tactical attributes (9 per team): `buildUpPlaySpeed`, `buildUpPlayDribbling`, `buildUpPlayPassing`, `chanceCreationPassing`, `chanceCreationCrossing`, `chanceCreationShooting`, `defencePressure`, `defenceAggression`, `defenceTeamWidth`.
3. **Preprocessing** — Median imputation for missing values, stratified 80/20 train-test split.
4. **Model** — Random Forest with `class_weight='balanced'` to correct Home Win class imbalance.
5. **Explainability** — SHAP TreeExplainer applied to 500 test samples to produce global (Bar, Beeswarm) and local (Waterfall) explanations.

---

## 📊 Key Findings

- `home_defencePressure` (0.0667) and `away_defencePressure` (0.0651) are the two most important features — defensive pressing intensity from **both** sides is the strongest predictor of match outcome.
- High `home_chanceCreationShooting` and `home_buildUpPlayPassing` push predictions toward **Home Win**, while high `away_defencePressure` pushes against it.
- Results are comparable to the interpretable framework proposed by Yeung et al. (2023, *PLOS ONE*), which achieved an F1-score of 0.47 using player-level FIFA ratings — suggesting team tactical style alone carries substantial predictive signal.

---

## 📚 Reference

Yeung, C. K., Bunker, R. P., & Fujii, K. (2023). A framework of interpretable match results prediction in football with FIFA ratings and team formation. *PLOS ONE, 18*(4), e0284318. https://doi.org/10.1371/journal.pone.0284318

---

## 📄 License

This project is intended for academic purposes (Riset Teknologi Informasi coursework). Dataset is publicly available on Kaggle under its respective license.

---

## 👤 Author

**[Your Name]**
[Your Study Program], [Your University]
