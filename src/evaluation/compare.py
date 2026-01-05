# ============================================================
# Compare Ranking Model: WITHOUT vs WITH Grid Position
# ============================================================
# Purpose:
# - Quantify the effect of race-relative grid position
# - Maintain strict leakage safety
# - Produce defensible Top-1 / Top-3 metrics
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBRanker

# ------------------------------------------------------------
# 1. Add race-relative (SAFE) features
# ------------------------------------------------------------
def add_race_relative_features(df):
    df = df.copy()

    # Qualifying features
    df["quali_gap_to_best"] = (
        df["best_quali_lap"]
        - df.groupby("race")["best_quali_lap"].transform("min")
    )

    df["quali_zscore"] = (
        df.groupby("race")["best_quali_lap"]
          .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    )

    # FP2 features
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

    # ✅ SAFE GRID FEATURE (race-relative)
    df["grid_gap_to_front"] = (
        df["grid_position"]
        - df.groupby("race")["grid_position"].transform("min")
    )

    return df


# ------------------------------------------------------------
# 2. Feature sets
# ------------------------------------------------------------
FEATURES_NO_GRID = [
    "fp2_gap_to_best",
    "fp2_zscore",
    "fp2_std_lap",
    "fp2_deg_zscore",
    "quali_gap_to_best",
    "quali_zscore",
    "best_quali_lap_number",
]

FEATURES_WITH_GRID = FEATURES_NO_GRID + ["grid_gap_to_front"]


# ------------------------------------------------------------
# 3. Training & evaluation function
# ------------------------------------------------------------
def train_and_eval(train_df, test_df, features, label):

    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    group_train = train_df.groupby("race").size().values

    model.fit(
        train_df[features],
        -train_df["position"],  # lower finishing position = better
        group=group_train,
    )

    test_df = test_df.copy()
    test_df["score"] = model.predict(test_df[features])

    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    test_df["win_prob"] = (
        test_df.groupby("race")["score"].transform(softmax)
    )

    top1 = 0
    top3 = 0
    total = 0

    for race, g in test_df.groupby("race"):
        total += 1
        g = g.sort_values("win_prob", ascending=False)

        winner = g.loc[g["position"] == 1, "driver"].values[0]

        if g.iloc[0]["driver"] == winner:
            top1 += 1
        if winner in g.head(3)["driver"].tolist():
            top3 += 1

    print(f"\n=== {label} ===")
    print(f"Top-1 Accuracy: {top1 / total:.4f}")
    print(f"Top-3 Accuracy: {top3 / total:.4f}")

    return {
        "model": label,
        "top1": top1 / total,
        "top3": top3 / total,
    }


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------
train_df = pd.read_csv("dataset_2022.csv")
test_df = pd.read_csv("dataset_2023.csv")

train_df = add_race_relative_features(train_df)
test_df = add_race_relative_features(test_df)


# ------------------------------------------------------------
# 5. Run comparison
# ------------------------------------------------------------
results = []

results.append(
    train_and_eval(
        train_df,
        test_df,
        FEATURES_NO_GRID,
        "Ranking Model (NO GRID)",
    )
)

results.append(
    train_and_eval(
        train_df,
        test_df,
        FEATURES_WITH_GRID,
        "Ranking Model (WITH GRID)",
    )
)

# ------------------------------------------------------------
# 6. Summary table
# ------------------------------------------------------------
summary = pd.DataFrame(results)

print("\n================= COMPARISON SUMMARY =================")
print(summary)
print("======================================================")
