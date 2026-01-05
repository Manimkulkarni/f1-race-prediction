import json
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\manim\OneDrive\Desktop\f1 race prediction\2024")
OUT = Path("fp2_laptimes_2024.csv")

rows = []

for race_dir in BASE.iterdir():
    if not race_dir.is_dir():
        continue

    fp2_dir = race_dir / "Practice 2"
    if not fp2_dir.exists():
        continue

    race_name = race_dir.name

    for driver_dir in fp2_dir.iterdir():
        if not driver_dir.is_dir():
            continue

        lap_file = driver_dir / "laptimes.json"
        if not lap_file.exists():
            continue

        driver = driver_dir.name

        with open(lap_file, "r") as f:
            data = json.load(f)

        times = data.get("time", [])
        laps = data.get("lap", [])
        compounds = data.get("compound", [])
        stints = data.get("stint", [])
        life = data.get("life", [])

        n = min(len(times), len(laps), len(compounds), len(stints), len(life))

        for i in range(n):
            lap_time = times[i]

            # Skip invalid lap times
            if lap_time in ("None", None):
                continue

            lap_no = laps[i]
            stint = stints[i]
            tyre_life = life[i]

            # Guard numeric fields
            if lap_no in ("None", None):
                continue
            if stint in ("None", None):
                continue
            if tyre_life in ("None", None):
                continue

            rows.append({
                "season": 2024,
                "race": race_name,
                "driver": driver,
                "lap_number": int(lap_no),
                "lap_time": float(lap_time),
                "compound": compounds[i],
                "stint": int(stint),
                "tyre_life": int(tyre_life),
            })

df = pd.DataFrame(rows)

df.to_csv(OUT, index=False)

print(f"Saved FP2 lap data → {OUT} ({len(df)} rows)")
