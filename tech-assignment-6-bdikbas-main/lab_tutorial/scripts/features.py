"""Feature Extraction

Uses raw 64 pixel values + one spatial feature (largest_blob via BFS).
Total: 65 features per sample.

Usage:
  uv run scripts/features.py
"""

import argparse
import numpy as np
import pandas as pd
from collections import deque

PIXEL_COLS = [f"pixel_{i}" for i in range(64)]


def _largest_connected_component(grid, threshold):
    """BFS to find largest connected region of pixels > threshold in 8x8 grid."""
    visited = [[False] * 8 for _ in range(8)]
    largest = 0
    for r in range(8):
        for c in range(8):
            if visited[r][c] or grid[r][c] <= threshold:
                continue
            size = 0
            q = deque([(r, c)])
            visited[r][c] = True
            while q:
                cr, cc = q.popleft()
                size += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and not visited[nr][nc] and grid[nr][nc] > threshold:
                        visited[nr][nc] = True
                        q.append((nr, nc))
            if size > largest:
                largest = size
    return largest


def _compute_blob_feature(raw_row):
    """Compute largest blob size for a single 64-element raw pixel row."""
    grid = raw_row.reshape(8, 8)
    median = np.median(raw_row)
    threshold = median + 3.0
    return float(_largest_connected_component(grid, threshold))


def engineer_features(df):
    """Extract features from cleaned DataFrame.

    64 raw pixel values + 1 BFS blob feature = 65 features.
    """
    pixels = df[PIXEL_COLS].values.astype(np.float32)

    blob = np.array([_compute_blob_feature(pixels[i]) for i in range(len(pixels))])
    blob = blob.reshape(-1, 1)

    X = np.hstack([pixels, blob])
    y = (df["label"].values == "present").astype(np.float32)
    groups = df["student_id"].values

    print(f"Features: {X.shape[1]} (64 raw pixels + 1 blob), "
          f"{len(X)} samples ({int(y.sum())} present, {int(len(y) - y.sum())} empty)")

    return X, y, groups


if __name__ == "__main__":
    from clean import clean_data

    parser = argparse.ArgumentParser(description="Extract features from cleaned data")
    parser.add_argument("--input", default="thermal_dataset.csv")
    parser.add_argument("--save", default=None, help="Save features to .npz file")
    args = parser.parse_args()

    df_clean = clean_data(args.input)
    X, y, groups = engineer_features(df_clean)

    if args.save:
        np.savez(args.save, X=X, y=y, groups=groups)
        print(f"Saved features to {args.save}")
