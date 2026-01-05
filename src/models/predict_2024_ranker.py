import pandas as pd
import numpy as np
from xgboost import XGBRanker

# =========================
# CONFIG
# =========================
TRAIN_SEASON = 2022
PREDICT_SEASON = 2024

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
pred_df  = pd.read_csv(f"dataset_{PREDICT_SEASON}.csv")

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
pred_df  = add_race_relative_features(pred_df)

# =========================
# RANKING TARGET (TRAIN)
# =========================
y_train = -train_df["position"]
group_train = train_df.groupby("race").size().values

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
# PREDICT 2024
# =========================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

pred_df["score"] = model.predict(pred_df[FEATURES])
pred_df["win_prob"] = (
    pred_df.groupby("race")["score"]
           .transform(softmax)
)

# =========================
# TOP-1 + TOP-3 PER RACE
# =========================
rows = []

for race, g in pred_df.groupby("race"):
    g = g.sort_values("win_prob", ascending=False)

    rows.append({
        "season": PREDICT_SEASON,
        "race": race,
        "predicted_winner": g.iloc[0]["driver"],
        "win_probability": g.iloc[0]["win_prob"],
        "top3": ", ".join(g.head(3)["driver"].tolist())
    })

pred_winners = pd.DataFrame(rows)

# =========================
# SAVE OUTPUTS
# =========================
pred_df[["season", "race", "driver", "win_prob"]].to_csv(
    "ranked_probs_2024.csv",
    index=False
)

pred_winners.to_csv(
    "ranked_winners_2024.csv",
    index=False
)

print("\n=== RANKING MODEL — PREDICTED WINNERS (2024) ===")
print(pred_winners)
print("\nSaved:")
print("→ ranked_probs_2024.csv")
print("→ ranked_winners_2024.csv")
