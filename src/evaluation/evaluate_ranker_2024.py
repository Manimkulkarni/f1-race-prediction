# ============================================================
# Ranker Evaluation Script — 2024 F1 Season (FINAL FIXED)
# ============================================================

import pandas as pd

# -----------------------------
# Step 1: Load real race winners
# -----------------------------
real = pd.read_csv("f1_2024_winners.csv")

print("Real results preview:")
print(real.head())
print("\nReal results columns:")
print(real.columns)

# Expected columns:
# - race
# - actual_winner

# -----------------------------
# Step 2: Load ranking predictions
# -----------------------------
pred = pd.read_csv("ranked_winners_2024.csv")

print("\nPredictions preview:")
print(pred.head())

# Expected columns:
# - season
# - race
# - predicted_winner
# - win_probability
# - top3

# -----------------------------
# Step 3: Merge predictions with real results
# -----------------------------
eval_df = pred.merge(real, on="race", how="inner")

# -----------------------------
# Step 4: Compute Top-1 correctness
# -----------------------------
eval_df["top1_correct"] = (
    eval_df["predicted_winner"] == eval_df["actual_winner"]
)

# -----------------------------
# Step 5: Compute Top-3 correctness
# -----------------------------
eval_df["top3_list"] = eval_df["top3"].apply(
    lambda x: [d.strip() for d in x.split(",")]
)

eval_df["top3_correct"] = eval_df.apply(
    lambda r: r["actual_winner"] in r["top3_list"],
    axis=1
)

# -----------------------------
# Step 6: Final evaluation table
# -----------------------------
final_table = eval_df[
    [
        "race",
        "predicted_winner",
        "actual_winner",
        "top3_list",
        "top1_correct",
        "top3_correct",
    ]
]

print("\nFinal evaluation table:")
print(final_table)

# -----------------------------
# Step 7: Compute final metrics
# -----------------------------
top1_acc = eval_df["top1_correct"].mean()
top3_acc = eval_df["top3_correct"].mean()

print("\n================= FINAL METRICS =================")
print(f"Top-1 Accuracy (2024, Ranker): {top1_acc:.3f}")
print(f"Top-3 Accuracy (2024, Ranker): {top3_acc:.3f}")
print(f"Top-1 Hits: {eval_df['top1_correct'].sum()} / {len(eval_df)}")
print(f"Top-3 Hits: {eval_df['top3_correct'].sum()} / {len(eval_df)}")
print("=================================================")

# -----------------------------
# Step 8: Save evaluation results
# -----------------------------
eval_df.to_csv("ranker_eval_2024.csv", index=False)
print("\nSaved → ranker_eval_2024.csv")
