import json
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\manim\OneDrive\Desktop\f1 race prediction\2024")
OUT = Path("race_results_2024.csv")

rows = []

for race_dir in BASE.iterdir():
    if not race_dir.is_dir():
        continue

    race = race_dir.name
    race_path = race_dir / "Race"

    if not race_path.exists():
        continue

    for driver_dir in race_path.iterdir():
        lap_file = driver_dir / "laptimes.json"
        if not lap_file.exists():
            continue

        driver = driver_dir.name

        with open(lap_file) as f:
            data = json.load(f)

        pos = data.get("pos", [])
        status = data.get("status", [])

        final_pos = None
        dnf = 0

        for i in range(len(pos)):
            if pos[i] not in ("None", None):
                final_pos = int(pos[i])

            if i < len(status) and status[i] not in ("1", 1):
                dnf = 1

        if final_pos is None:
            continue

        rows.append({
            "season": 2024,
            "race": race,
            "driver": driver,
            "position": final_pos,
            "win": 1 if final_pos == 1 else 0,
            "dnf": dnf
        })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"Saved race results → {OUT} ({len(df)} rows)")
