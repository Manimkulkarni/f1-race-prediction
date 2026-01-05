import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_FILES = ["dataset_2022.csv", "dataset_2023.csv"]
PREDICT_FILE = "dataset_2024.csv"
WINNERS_FILE = "f1_2024_winners.csv"  # actual winners CSV

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
# FUNCTIONS
# -----------------------------
def race_relative_normalize(df, feature_cols):
    df = df.copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mean = df.groupby("race")[col].transform("mean")
        std = df.groupby("race")[col].transform("std")
        df[col] = np.where(std == 0, 0.0, (df[col] - mean) / std)
    return df

def racewise_softmax(df, score_col):
    df = df.copy()
    df["win_prob"] = 0.0
    for race, g in df.groupby("race"):
        scores = g[score_col].values
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        df.loc[g.index, "win_prob"] = probs
    return df

# -----------------------------
# LOAD & TRAIN
# -----------------------------
train_df = pd.concat([pd.read_csv(f) for f in TRAIN_FILES])
train_df = race_relative_normalize(train_df, REL_FEATURES)

X_train = train_df[FEATURES]
y_train = train_df["win"]

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)

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
# LOAD 2024 DATA
# -----------------------------
pred_df = pd.read_csv(PREDICT_FILE)
pred_df = race_relative_normalize(pred_df, REL_FEATURES)

X_pred = pred_df[FEATURES]
X_pred_imp = imputer.transform(X_pred)

# -----------------------------
# PREDICT PROBABILITIES
# -----------------------------
pred_df["score"] = model.predict_proba(X_pred_imp)[:, 1]
pred_df = racewise_softmax(pred_df, "score")



# Save full table with probabilities
pred_df.to_csv("predicted_probs_2024.csv", index=False)

# Print top-3 per race
for race, g in pred_df.groupby("race"):
    g_sorted = g.sort_values("win_prob", ascending=False)
    print(f"\n=== {race} ===")
    print(g_sorted[["driver", "win_prob"]].head(3))

# -----------------------------
# PRINT TOP-3 PER RACE
# -----------------------------
for race, g in pred_df.groupby("race"):
    g_sorted = g.sort_values("win_prob", ascending=False)
    print(f"\n=== {race} ===")
    print(g_sorted[["driver", "win_prob"]].head(3))

# -----------------------------
# SAVE PREDICTED WINNERS CSV
# -----------------------------
winners = []
for race, g in pred_df.groupby("race"):
    g_sorted = g.sort_values("win_prob", ascending=False)
    winners.append({
        "season": 2024,
        "race": race,
        "predicted_winner": g_sorted.iloc[0]["driver"],
        "win_probability": g_sorted.iloc[0]["win_prob"]
    })

out_df = pd.DataFrame(winners)
out_df.to_csv("predicted_winners_2024.csv", index=False)

# -----------------------------
# LOAD ACTUAL WINNERS
# -----------------------------
real = pd.read_csv(WINNERS_FILE)
real["race"] = real["race"].str.strip()
real = real.set_index("race")

# -----------------------------
# TOP-1 & TOP-3 ACCURACY FROM FULL TABLE
# -----------------------------
top1_correct = 0
top3_correct = 0
total_races = pred_df["race"].nunique()

for race, g in pred_df.groupby("race"):
    g_sorted = g.sort_values("win_prob", ascending=False)
    top3_drivers = g_sorted.iloc[:3]["driver"].tolist()
    winner = real.loc[race, "actual_winner"]
    if g_sorted.iloc[0]["driver"] == winner:
        top1_correct += 1
    if winner in top3_drivers:
        top3_correct += 1

top1 = top1_correct / total_races
top3 = top3_correct / total_races

print("\n=== TOP ACCURACY ===")
print(f"Top-1 Accuracy: {top1:.4f}")
print(f"Top-3 Accuracy: {top3:.4f}")
