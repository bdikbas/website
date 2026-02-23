"""Data Cleaning (provided complete — no TODOs)

Same as Challenge 1. Import your features.py from Challenge 1 to use with this.
"""

import numpy as np
import pandas as pd

PIXEL_COLS = [f"pixel_{i}" for i in range(64)]


def clean_data(csv_path="thermal_dataset.csv"):
    df = pd.read_csv(csv_path)
    pixels = df[PIXEL_COLS].values.astype(np.float32)
    labels = df["label"].values

    reasons = []
    for i in range(len(df)):
        row = pixels[i]
        label = labels[i]
        row_min = row.min()
        row_max = row.max()
        row_range = row_max - row_min
        row_median = np.median(row)
        elevation = row_max - row_median

        if row_min < 5.0 or row_max > 50.0:
            reasons.append("sensor_glitch")
        elif label == "present" and row_range < 0.5:
            reasons.append("flat_present")
        elif label == "present" and elevation < 2.0:
            reasons.append("ambiguous_present")
        elif label == "empty" and elevation > 8.0:
            reasons.append("ambiguous_empty")
        else:
            reasons.append("")

    df["reason"] = reasons
    bad = df[df["reason"] != ""].copy()
    good = df[df["reason"] == ""].drop(columns=["reason"]).copy()

    print(f"Cleaning: {len(df)} rows -> {len(good)} kept, {len(bad)} discarded")
    for reason, count in bad["reason"].value_counts().items():
        print(f"  {reason}: {count}")

    return good
