import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def compute_degradation(fp2_csv, out_csv):
    df = pd.read_csv(fp2_csv)

    rows = []

    for (season, race, driver), g in df.groupby(["season", "race", "driver"]):
        slopes = []

        for stint, stint_df in g.groupby("stint"):
            if len(stint_df) < 5:
                continue

            X = stint_df[["tyre_life"]].values
            y = stint_df["lap_time"].values

            model = LinearRegression()
            model.fit(X, y)

            slopes.append(model.coef_[0])

        if len(slopes) == 0:
            continue

        rows.append({
            "season": season,
            "race": race,
            "driver": driver,
            "fp2_deg_slope": float(np.mean(slopes))
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    print(f"Saved FP2 degradation → {out_csv}")

# Run for each season
#compute_degradation("fp2_laptimes_2022.csv", "fp2_deg_2022.csv")
#compute_degradation("fp2_laptimes_2023.csv", "fp2_deg_2023.csv")
compute_degradation("fp2_laptimes_2024.csv", "fp2_deg_2024.csv")