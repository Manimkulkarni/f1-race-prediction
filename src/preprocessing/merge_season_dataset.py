import pandas as pd

def build_season_dataset(season):
    fp2 = pd.read_csv(f"fp2_features_{season}.csv")
    quali = pd.read_csv(f"qualifying_{season}.csv")
    race = pd.read_csv(f"race_results_{season}.csv")
    deg = pd.read_csv(f"fp2_deg_{season}.csv")

    # 1️⃣ Merge FP2 + Qualifying
    df = fp2.merge(
        quali,
        on=["season", "race", "driver"],
        how="inner"
    )

    # 2️⃣ Merge FP2 degradation (LEFT join – some drivers may not have long runs)
    df = df.merge(
        deg,
        on=["season", "race", "driver"],
        how="left"
    )

    # 3️⃣ Merge Race results (labels)
    df = df.merge(
        race,
        on=["season", "race", "driver"],
        how="inner"
    )

    out = f"dataset_{season}.csv"
    df.to_csv(out, index=False)

    print(f"Built dataset → {out}")
    print(f"Rows: {len(df)}")

# Build datasets
#build_season_dataset(2022)
#build_season_dataset(2023)
build_season_dataset(2024)