"""Challenge 1: Feature Engineering

Implement per-sample normalization, intensity statistics, and spatial
features for the AMG8833 thermal sensor data.

Features to implement (76 total):
  REQUIRED
  --------
  64 normalized pixels: (pixel - median) / std          [Part A]
  4 intensity stats:    row_max, range, count_above_3,  [Part B]
                        count_above_5
  spatial_gradient      (TODO A)
  largest_blob          (TODO B)
  quadrant_var          (TODO C)
  center_vs_edge        (TODO D)
  row_profile_std       (TODO E)
  col_profile_std       (TODO E)

  EXTRA CREDIT
  ------------
  hot_centroid_r        (TODO F)
  hot_pixel_ratio       (TODO F)

  Unimplemented EC features default to 0.0, which is fine —
  your model will still train and run on the ESP32.

Note — looking ahead to Challenge 2:
  In Challenge 2 you will train a neural network using these features.
  You will use L2 regularization (required) to prevent overfitting by
  penalizing large weights. As extra credit you can also explore L1
  regularization (which drives some weights to zero) and Dropout
  (which randomly disables neurons during training to improve generalization).

Usage:
  uv run scripts/features.py
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
from clean import clean_data

PIXEL_COLS = [f"pixel_{i}" for i in range(64)]


def _largest_connected_component(grid, threshold):
    """Find the largest connected region of pixels > threshold in an 8x8 grid.

    Use BFS (breadth-first search) with 4-directional connectivity
    (up, down, left, right — no diagonals).

    Args:
        grid: 8x8 numpy array of temperature values.
        threshold: Pixels must be strictly greater than this to be "hot".

    Returns:
        int: Size of the largest connected component of hot pixels.
             Returns 0 if no pixels exceed the threshold.

    Hint:
        - Use a visited[][] array to track which cells you've already checked
        - For each unvisited hot pixel, do a BFS to find all connected hot pixels
        - Track the size of each component, keep the largest
        - The grid is small (8x8 = 64 cells) so efficiency doesn't matter
    """
    # TODO: Implement BFS to find the largest connected component
    # The basic algorithm:
    #   1. Create a visited array (8x8, all False)
    #   2. For each cell (r, c) in the grid:
    #      - If already visited or grid[r][c] <= threshold, skip
    #      - Otherwise, start a BFS from (r, c):
    #        a. Create a queue, add (r, c), mark visited
    #        b. While queue not empty:
    #           - Pop (cr, cc), increment size counter
    #           - Check 4 neighbors (up/down/left/right)
    #           - If neighbor is in-bounds, not visited, and > threshold:
    #             mark visited and add to queue
    #      - Update largest if this component's size is bigger
    #   3. Return largest
    visited = np.zeros((8, 8), dtype=bool)
    largest = 0

    for r in range(8):
        for c in range(8):
            if visited[r, c] or grid[r, c] <= threshold:
                continue

            # BFS from (r, c)
            q = deque()
            q.append((r, c))
            visited[r, c] = True
            size = 0

            while q:
                cr, cc = q.popleft()
                size += 1

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        if not visited[nr, nc] and grid[nr, nc] > threshold:
                            visited[nr, nc] = True
                            q.append((nr, nc))

            if size > largest:
                largest = size

    return int(largest)


def compute_spatial_features(raw_row):
    """Compute 8 spatial features from a single 64-element raw pixel row.

    Args:
        raw_row: numpy array of shape (64,) — raw temperature values.

    Returns:
        numpy array of shape (8,) containing:
            [spatial_gradient, largest_blob, quadrant_var, center_vs_edge,
             row_profile_std, col_profile_std, hot_centroid_r, hot_pixel_ratio]

        REQUIRED: spatial_gradient (A), largest_blob (B),
                  quadrant_var (C), center_vs_edge (D),
                  row_profile_std / col_profile_std (E)
        EXTRA CREDIT: hot_centroid_r / hot_pixel_ratio (F)
        Unimplemented features default to 0.0.
    """
    grid = raw_row.reshape(8, 8)
    median = np.median(raw_row)
    threshold = median + 3.0

    # TODO A [REQUIRED]: spatial_gradient
    # Compute the mean absolute difference between horizontal neighbors
    # (grid[:, 1:] vs grid[:, :-1]) and vertical neighbors
    # (grid[1:, :] vs grid[:-1, :]). Average the two means.
    # This measures how "sharp" the temperature transitions are.
    horiz = np.abs(grid[:, 1:] - grid[:, :-1]).mean()  # 8x7 diffs
    vert  = np.abs(grid[1:, :] - grid[:-1, :]).mean()  # 7x8 diffs
    spatial_gradient = 0.5 * (horiz + vert)

    # TODO B [REQUIRED]: largest_blob
    # Use your _largest_connected_component function to find the largest
    # connected region of pixels > threshold (4-directional connectivity).
    # A person creates a contiguous warm region; random noise does not.
    largest_blob = _largest_connected_component(grid, threshold)

    # TODO C [REQUIRED]: quadrant_var
    # Split the 8x8 grid into four 4x4 quadrants:
    #   top-left:     grid[:4, :4]
    #   top-right:    grid[:4, 4:]
    #   bottom-left:  grid[4:, :4]
    #   bottom-right: grid[4:, 4:]
    # Compute the mean of each quadrant (4 values).
    # Return the variance of those 4 means.
    # This detects if heat is concentrated in one area vs spread evenly.
    q_means = np.array([
        grid[:4, :4].mean(),
        grid[:4, 4:].mean(),
        grid[4:, :4].mean(),
        grid[4:, 4:].mean(),
    ], dtype=np.float32)
    quadrant_var = float(np.var(q_means))


    # TODO D [REQUIRED]: center_vs_edge
    # Compute: mean(center 4x4 region) - mean(outer ring pixels)
    # Center region: rows 2-5, cols 2-5 (i.e., grid[2:6, 2:6])
    # Outer ring: all other pixels (the border)
    # Hint: Use a boolean mask — True for outer, False for center.
    center = grid[2:6, 2:6]
    center_mean = center.mean()

    outer_mask = np.ones((8, 8), dtype=bool)
    outer_mask[2:6, 2:6] = False
    edge_mean = grid[outer_mask].mean()
    center_vs_edge = float(center_mean - edge_mean)

    # TODO E [REQUIRED]: row_profile_std and col_profile_std
    # Take the max value in each row → 8 values → compute their std.
    # Do the same for columns.
    # If a person is present, the row/column maxima will vary more
    # (some rows/cols have the person, others don't).
    row_profile_std = float(np.std(grid.max(axis=1)))
    col_profile_std = float(np.std(grid.max(axis=0)))

    # TODO F [EXTRA CREDIT]: hot_centroid_r and hot_pixel_ratio
    # Find all pixels > threshold ("hot pixels").
    # Compute the centroid (mean row, mean col) of the hot pixels.
    # hot_centroid_r = euclidean distance from centroid to grid center (3.5, 3.5)
    # hot_pixel_ratio = count of hot pixels / 64
    # If no hot pixels exist, set hot_centroid_r = 0.
    #
    # Hint: np.where(grid > threshold) returns (row_indices, col_indices)
    hot_r, hot_c = np.where(grid > threshold)
    hot_count = hot_r.size
    hot_pixel_ratio = float(hot_count / 64.0)

    if hot_count == 0:
        hot_centroid_r = 0.0
    else:
        hot_centroid_r = np.sqrt(np.mean((hot_r - 3.5)**2 + (hot_c - 3.5)**2))


    return np.array([
        spatial_gradient, largest_blob, quadrant_var, center_vs_edge,
        row_profile_std, col_profile_std, hot_centroid_r, hot_pixel_ratio,
    ], dtype=np.float32)


def engineer_features(df):
    """Engineer all 76 features from cleaned DataFrame.

    Args:
        df: Cleaned DataFrame with pixel_0..pixel_63 and label columns.

    Returns:
        tuple: (X, y) where X has shape (n_samples, 76) and y has shape (n_samples,)
    """
    pixels = df[PIXEL_COLS].values.astype(np.float32)

    # === Part A: Ambient Normalization (64 features) ===
    # The AMG8833 readings vary with room temperature (ambient).
    # A 25C room and a 30C room produce completely different raw values
    # for the same scene. Normalize each sample relative to its own
    # ambient baseline so features are comparable across environments.
    #
    # TODO: Compute per-row median and std across the 64 pixels.
    #       Then normalize: (pixel - median) / std
    #       Guard against division by zero: clamp std to min 0.1
    #
    # Hint:
    #   medians = np.median(pixels, axis=1, keepdims=True)  # shape: (n_samples, 1)
    #   stds = np.std(pixels, axis=1, keepdims=True)        # shape: (n_samples, 1)
    #   stds[stds < 0.1] = 0.1  # prevent division by zero
    #   normalized = (pixels - medians) / stds              # shape: (n_samples, 64)
    medians = np.median(pixels, axis=1, keepdims=True)
    stds = np.std(pixels, axis=1, keepdims=True)
    stds[stds < 0.1] = 0.1
    normalized = (pixels - medians) / stds

    # === Part B: Intensity Statistics (4 features) ===
    # These capture "how hot is the hottest pixel" and "how many pixels
    # are significantly above ambient" — signals that a warm body is present.
    #
    # TODO: Compute the following (each should be shape (n_samples, 1)):
    #   row_max:       maximum pixel value in each row (sample)
    #   row_range:     max - min pixel value in each row
    #   count_above_3: number of pixels more than 3 degrees above median
    #   count_above_5: number of pixels more than 5 degrees above median
    #
    # Hint: Use axis=1, keepdims=True for row_max
    #   row_max = pixels.max(axis=1, keepdims=True)
    #   For counts, use boolean comparison:
    #     (pixels > (medians + 3.0)).sum(axis=1).reshape(-1, 1)
    row_max = pixels.max(axis=1, keepdims=True)
    row_range = (pixels.max(axis=1, keepdims=True) - pixels.min(axis=1, keepdims=True))
    count_above_3 = (pixels > (medians + 3.0)).sum(axis=1).reshape(-1, 1)
    count_above_5 = (pixels > (medians + 5.0)).sum(axis=1).reshape(-1, 1)

    # Part C: Spatial Features 
    spatial = np.array([compute_spatial_features(row) for row in pixels])

    # Putting everything together
    X = np.hstack([normalized, row_max, row_range, count_above_3, count_above_5, spatial])
    y = (df["label"].values == "present").astype(np.float32)

    print(f"Features: {X.shape[1]}, Samples: {len(X)} "
          f"({int(y.sum())} present, {int(len(y) - y.sum())} empty)")

    return X, y


if __name__ == "__main__":
    df_clean = clean_data("thermal_dataset.csv")
    X, y = engineer_features(df_clean)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Sample feature vector (first 5): {X[0, :5]}")
