import pandas as pd
import numpy as np
from xgboost import XGBRanker

# =========================
# CONFIG
# =========================
TRAIN_SEASON = 2022
TEST_SEASON = 2023

FEATURES = [
    "fp2_gap_to_best",
    "fp2_zscore",
    "fp2_std_lap",
    "fp2_deg_zscore",
    "quali_gap_to_best",
    "quali_zscore",
    "best_quali_lap_number"
]

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv(f"dataset_{TRAIN_SEASON}.csv")
test_df  = pd.read_csv(f"dataset_{TEST_SEASON}.csv")

# =========================
# RACE-RELATIVE FEATURES
# =========================
def add_race_relative_features(df):
    df = df.copy()

    df["quali_gap_to_best"] = (
        df["best_quali_lap"]
        - df.groupby("race")["best_quali_lap"].transform("min")
    )

    df["quali_zscore"] = (
        df.groupby("race")["best_quali_lap"]
          .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    )

    df["fp2_gap_to_best"] = (
        df["fp2_mean_lap"]
        - df.groupby("race")["fp2_mean_lap"].transform("min")
    )

    df["fp2_zscore"] = (
        df.groupby("race")["fp2_mean_lap"]
          .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    )

    df["fp2_deg_zscore"] = (
        df.groupby("race")["fp2_deg_slope"]
          .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    )

    return df

train_df = add_race_relative_features(train_df)
test_df  = add_race_relative_features(test_df)

# =========================
# RANKING TARGET
# =========================
# Lower position = better → invert
y_train = -train_df["position"]
y_test  = -test_df["position"]

group_train = train_df.groupby("race").size().values
group_test  = test_df.groupby("race").size().values

# =========================
# TRAIN RANKING MODEL
# =========================
model = XGBRanker(
    objective="rank:pairwise",
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(
    train_df[FEATURES],
    y_train,
    group=group_train
)

# =========================
# PREDICT + SOFTMAX
# =========================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

test_df["score"] = model.predict(test_df[FEATURES])
test_df["win_prob"] = (
    test_df.groupby("race")["score"]
           .transform(softmax)
)

# =========================
# EVALUATION
# =========================
top1_hits = 0
top3_hits = 0
total = 0

for race, g in test_df.groupby("race"):
    total += 1
    g = g.sort_values("win_prob", ascending=False)

    actual_winner = g.loc[g["position"] == 1, "driver"].values[0]
    top1 = g.iloc[0]["driver"]
    top3 = g.head(3)["driver"].tolist()

    if actual_winner == top1:
        top1_hits += 1
    if actual_winner in top3:
        top3_hits += 1

print("\n=== RANKING MODEL RESULTS (2023) ===")
print(f"Top-1 Accuracy : {top1_hits / total:.4f}")
print(f"Top-3 Accuracy : {top3_hits / total:.4f}")

# =========================
# SAVE OUTPUTS
# =========================
out_cols = ["season", "race", "driver", "win_prob", "position"]
test_df[out_cols].to_csv(
    f"ranked_predictions_{TEST_SEASON}.csv",
    index=False
)

print(f"\nSaved ranked predictions → ranked_predictions_{TEST_SEASON}.csv")
