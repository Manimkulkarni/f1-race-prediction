import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_FILE = "dataset_2022.csv"
TEST_FILE = "dataset_2023.csv"

TARGET = "win"

FEATURES = [
    "fp2_mean_lap",
    "fp2_std_lap",
    "fp2_best_lap",
    "fp2_laps_count",
    "fp2_deg_slope",
    "best_quali_lap",
    "best_quali_lap_number",
    "dnf"
]

REL_FEATURES = [
    "fp2_mean_lap",
    "fp2_std_lap",
    "fp2_best_lap",
    "fp2_laps_count",
    "fp2_deg_slope",
    "best_quali_lap",
    "best_quali_lap_number"
]

# -----------------------------
# RACE-RELATIVE NORMALIZATION
# -----------------------------
def race_relative_normalize(df, cols):
    df = df.copy()

    for col in cols:
        df[col] = df[col].astype(float)

    for race, g in df.groupby("race"):
        idx = g.index
        for col in cols:
            mean = g[col].mean()
            std = g[col].std()
            if std == 0 or pd.isna(std):
                df.loc[idx, col] = 0.0
            else:
                df.loc[idx, col] = (g[col] - mean) / std

    return df

# -----------------------------
# RACE-WISE SOFTMAX
# -----------------------------
def racewise_softmax(df, score_col):
    df = df.copy()
    df["win_prob_softmax"] = 0.0

    for race, g in df.groupby("race"):
        scores = g[score_col].values
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        df.loc[g.index, "win_prob_softmax"] = probs

    return df

# -----------------------------
# LOAD DATA
# -----------------------------
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# -----------------------------
# NORMALIZE (NO LEAKAGE)
# -----------------------------
train_df = race_relative_normalize(train_df, REL_FEATURES)
test_df = race_relative_normalize(test_df, REL_FEATURES)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# -----------------------------
# IMPUTATION
# -----------------------------
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# -----------------------------
# MODEL
# -----------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_imp, y_train)

# -----------------------------
# TRAIN METRICS
# -----------------------------
train_proba = model.predict_proba(X_train_imp)[:, 1]
train_pred = (train_proba > 0.5).astype(int)

print("\n=== TRAINING RESULTS (2022) ===")
print("Log Loss :", log_loss(y_train, train_proba))
print("Accuracy :", accuracy_score(y_train, train_pred))

# -----------------------------
# TEST PREDICTIONS
# -----------------------------
test_df = test_df.copy()
test_df["score"] = model.predict_proba(X_test_imp)[:, 1]

# Apply race-wise softmax
test_df = racewise_softmax(test_df, "score")

# -----------------------------
# TEST METRICS
# -----------------------------
test_logloss = log_loss(y_test, test_df["win_prob_softmax"])
test_accuracy = accuracy_score(
    y_test, (test_df["win_prob_softmax"] > 0.5).astype(int)
)

# Top-K Accuracy
top1 = 0
top3 = 0

for race, g in test_df.groupby("race"):
    g = g.sort_values("win_prob_softmax", ascending=False)
    winner_idx = g[g["win"] == 1].index[0]

    if winner_idx == g.index[0]:
        top1 += 1
    if winner_idx in g.index[:3]:
        top3 += 1

n_races = test_df["race"].nunique()
top1 /= n_races
top3 /= n_races

print("\n=== TEST RESULTS (2023) ===")
print("Log Loss :", test_logloss)
print("Accuracy :", test_accuracy)
print("Top-1 Accuracy :", top1)
print("Top-3 Accuracy :", top3)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
imp_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== Feature Importance (XGBoost) ===")
print(imp_df)
