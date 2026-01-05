import json
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\manim\OneDrive\Desktop\f1 race prediction\2024")
OUT = Path("qualifying_2024.csv")

rows = []

for race_dir in BASE.iterdir():
    if not race_dir.is_dir():
        continue

    quali_dir = race_dir / "Qualifying"
    if not quali_dir.exists():
        continue

    race = race_dir.name

    for driver_dir in quali_dir.iterdir():
        lap_file = driver_dir / "laptimes.json"
        if not lap_file.exists():
            continue

        driver = driver_dir.name

        with open(lap_file) as f:
            data = json.load(f)

        times = data.get("time", [])
        laps = data.get("lap", [])
        compounds = data.get("compound", [])

        best_time = None
        best_lap = None
        best_compound = None

        for i in range(len(times)):
            t = times[i]
            if t in ("None", None):
                continue

            t = float(t)

            if best_time is None or t < best_time:
                best_time = t
                best_lap = int(laps[i])
                best_compound = compounds[i]

        if best_time is None:
            continue

        rows.append({
            "season": 2024,
            "race": race,
            "driver": driver,
            "best_quali_lap": best_time,
            "best_quali_lap_number": best_lap,
            "best_quali_compound": best_compound
        })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"Saved qualifying data → {OUT} ({len(df)} rows)")
