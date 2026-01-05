import pandas as pd

def build_fp2_features(fp2_csv, out_csv):
    df = pd.read_csv(fp2_csv)

    features = (
        df
        .groupby(["season", "race", "driver"])
        .agg(
            fp2_mean_lap=("lap_time", "mean"),
            fp2_std_lap=("lap_time", "std"),
            fp2_best_lap=("lap_time", "min"),
            fp2_laps_count=("lap_time", "count"),
        )
        .reset_index()
    )

    features.to_csv(out_csv, index=False)
    print(f"Saved FP2 features → {out_csv}")

# Run for both seasons
#build_fp2_features("fp2_laptimes_2022.csv", "fp2_features_2022.csv")
build_fp2_features("fp2_laptimes_2023.csv", "fp2_features_2023.csv")
build_fp2_features("fp2_laptimes_2024.csv", "fp2_features_2024.csv")