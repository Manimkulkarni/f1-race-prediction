# f1-race-prediction
Pre-race ML model to predict Formula 1 race winners using FP2 and qualifying data
# 🏎️ Formula 1 Race Winner Prediction (Pre-Race ML)

This project builds a **strictly pre-race machine learning pipeline** to predict **Formula 1 race winners** using practice and qualifying data.  
The goal is to evaluate how far **publicly available, pre-race information** can go in predicting race outcomes—without using FP3, sprint races, or race-day incidents.

---

## 🎯 Problem Statement

Given only **pre-race information**, predict:
- The **most likely race winner** (Top-1)
- A shortlist of **likely contenders** (Top-3)

The project emphasizes:
- Clean feature engineering
- No data leakage
- Realistic evaluation on **future seasons**

---

## 🚫 Explicit Constraints (Very Important)

This project **does NOT use**:
- FP3 data
- Sprint race results
- Race incidents (safety cars, DNFs, penalties during race)
- In-race telemetry
- Post-race standings as features

Only **information known before race start** is allowed.

---

## 🧠 Modeling Approaches

### 1️⃣ Baseline Classification
- Logistic Regression
- XGBoost Classifier  
Predicts winner as a binary classification problem.

### 2️⃣ Pairwise Ranking Model (Main Contribution)
- **XGBoost Ranker (`rank:pairwise`)**
- Learns **relative driver competitiveness within each race**
- Converts ranking scores to **race-wise softmax probabilities**

This formulation significantly improves Top-1 accuracy compared to classification.

---

## 📊 Features Used (Pre-Race Only)

### Practice (FP2)
- Mean lap time
- Best lap
- Lap time standard deviation
- Tyre degradation slope

### Qualifying
- Best qualifying lap time
- Qualifying lap number
- Grid position (race-relative)
- Gap to pole (race-relative)

### Derived (Race-Relative)
- Z-scores within each race
- Grid gap to front

> All race-relative normalization is done **per race** to remove track-length bias.

📈 Results Summary
Model	Top-1 Accuracy	Top-3 Accuracy
Logistic Regression	~16%	~55%
XGBoost Classifier	~18–19%	~55–60%
XGBoost Ranker	~31%	~62%

On the 2024 season:

Correct winner predicted in ~1 out of 3 races

Actual winner appears in Top-3 predictions in ~60%+ of races

These results are strong for a strict pre-race setting.
📌 Key Takeaways

Ranking formulation > classification for race winner prediction

Race-relative features are crucial

Pre-race uncertainty is real and unavoidable

The model spreads probability mass realistically instead of over-confident picks

🔮 Possible Extensions

Pairwise ranking with listwise loss

Weather-aware modeling (forecast-based)

Team-level hierarchical models

Strategy simulation (out-of-scope for pre-race)
---

## 📁 Project Structure

f1-race-prediction/
│
├── data/
│ ├── raw/ # Raw telemetry (per season)
│ ├── processed/ # Feature tables & datasets
│ └── external/ # Static metadata & real results
│
├── src/
│ ├── extraction/ # FP2, qualifying, race extractors
│ ├── preprocessing/ # Dataset merging
│ ├── models/ # Training & prediction scripts
│ └── evaluation/ # Accuracy & comparison scripts
│
├── notebooks/
│ └── analysis_2024.ipynb # Visual analysis & plots
│
├── outputs/ # Model predictions & evaluations
│
├── README.md
└── requirements.txt
