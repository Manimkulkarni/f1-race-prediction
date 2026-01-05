import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

def race_relative_normalize(df, feature_cols):
    df = df.copy()

    for col in feature_cols:
        df[col] = df[col].astype(float)

    for race, race_df in df.groupby("race"):
        idx = race_df.index

        for col in feature_cols:
            mean = race_df[col].mean()
            std = race_df[col].std()

            if std == 0 or pd.isna(std):
                df.loc[idx, col] = 0.0
            else:
                df.loc[idx, col] = (race_df[col] - mean) / std

    return df


# -----------------------------
# Config
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

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

REL_FEATURES = [
    "fp2_mean_lap",
    "fp2_std_lap",
    "fp2_best_lap",
    "fp2_laps_count",
    "fp2_deg_slope",
    "best_quali_lap",
    "best_quali_lap_number",
]

train_df = race_relative_normalize(train_df, REL_FEATURES)
test_df = race_relative_normalize(test_df, REL_FEATURES)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# -----------------------------
# Pipeline (PROFESSIONAL)
# -----------------------------
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

# -----------------------------
# Train
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Predict probabilities
# -----------------------------
train_probs = pipeline.predict_proba(X_train)[:, 1]
test_probs = pipeline.predict_proba(X_test)[:, 1]

# -----------------------------
# Metrics
# -----------------------------
train_logloss = log_loss(y_train, train_probs)
test_logloss = log_loss(y_test, test_probs)

train_acc = accuracy_score(y_train, train_probs > 0.5)
test_acc = accuracy_score(y_test, test_probs > 0.5)

# -----------------------------
# Top-K Accuracy
# -----------------------------
def top_k_accuracy(df, probs, k=3):
    df = df.copy()
    df["prob"] = probs

    correct = 0
    total = 0

    for race, race_df in df.groupby("race"):
        top_k = race_df.sort_values("prob", ascending=False).head(k)
        if top_k["win"].sum() > 0:
            correct += 1
        total += 1

    return correct / total

top1 = top_k_accuracy(test_df, test_probs, k=1)
top3 = top_k_accuracy(test_df, test_probs, k=3)

# -----------------------------
# Output
# -----------------------------
print("\n=== TRAINING RESULTS (2022) ===")
print(f"Log Loss : {train_logloss:.4f}")
print(f"Accuracy : {train_acc:.4f}")

print("\n=== TEST RESULTS (2023) ===")
print(f"Log Loss : {test_logloss:.4f}")
print(f"Accuracy : {test_acc:.4f}")
print(f"Top-1 Accuracy : {top1:.4f}")
print(f"Top-3 Accuracy : {top3:.4f}")

# -----------------------------
# Feature importance
# -----------------------------
coef = pipeline.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": coef
}).sort_values("coefficient", key=abs, ascending=False)

print("\n=== Feature Importance (LogReg Coefficients) ===")
print(coef_df.to_string(index=False))
