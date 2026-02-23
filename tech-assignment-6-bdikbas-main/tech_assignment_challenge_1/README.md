# Tech Assignment Challenge 1 — Feature Engineering

Implement feature engineering for the AMG8833 thermal sensor data. You will transform
raw 8x8 temperature grids into 76 meaningful features that a neural network can learn from.

## Why not just use raw pixels?

- Different rooms = different ambient temps
- A person shows up as a *localized, contiguous warm blob* — not just "some hot pixels"
- Spatial structure of the heat pattern matters

You'll implement three feature groups:

- **Normalized pixels (64)** — ambient-invariant pixel values
- **Intensity stats (4)** — how hot, how spread out, how many hot pixels
- **Spatial features (8)** — gradients, blob size, centroid, quadrant distribution

## Your task

Open `scripts/features.py` and complete the TODOs.

**Part A — Ambient normalization (64 features)** ⬅ REQUIRED
For each sample: median and std of the 64 pixels, then normalize with `(pixel - median) / std`. A 25C room and 30C room should give similar feature values for the same scene.

**Part B — Intensity stats (4 features)** ⬅ REQUIRED
- `row_max` — max pixel temp
- `row_range` — max - min
- `count_above_3` — pixels more than 3C above median
- `count_above_5` — pixels more than 5C above median

**Part C — Spatial features (8 features)**

| Feature | Description | Status |
|---|---|---|
| `spatial_gradient` | sharpness of temp transitions | **REQUIRED** |
| `largest_blob` | size of largest connected hot region (BFS) | **REQUIRED** |
| `quadrant_var` | how unevenly heat is distributed across quadrants | **REQUIRED** |
| `center_vs_edge` | center warmth vs edge warmth | **REQUIRED** |
| `row_profile_std`, `col_profile_std` | variation in per-row/col max temps | **REQUIRED** |
| `hot_centroid_r` | distance of hot-pixel centroid from grid center | *Extra Credit* |
| `hot_pixel_ratio` | fraction of pixels above threshold | *Extra Credit* |

Unimplemented extra credit features default to `0.0` — your model will still train and run correctly without them.

The BFS for `largest_blob` is the trickiest required part. Think about it as: for each unvisited "hot" pixel, explore all connected hot neighbors (up/down/left/right) and count the region size.

> **Looking ahead to Challenge 2:** In Challenge 2 you'll train a neural network on these 76 features. You'll use **L2 regularization** (required) — which penalizes large weights to prevent overfitting — as well as **L1 regularization** (drives some weights to exactly zero, like a feature selector) and **Dropout** (randomly disables neurons during training to improve generalization) as extra credit.

## Run the pipeline

```bash
uv run scripts/features.py
```

When done you should see something like:
```
Cleaning: XXXX rows -> XXXX kept, XX discarded
Features: 76, Samples: XXXX (XXX present, XXX empty)
Feature matrix shape: (XXXX, 76)
```

## Tests

Tests live in `tests/test_features.py` and check your implementation against known inputs.

**Run required tests only:**
```bash
uv run -m pytest tests/test_features.py -v
```

**Run required + extra credit tests:**
```bash
uv run -m pytest tests/test_features.py -v --run-ec
```

What each group tests:

| Test class | What it checks |
|---|---|
| `TestNormalization` | Normalized output has 76 features; mean near 0; ambient-invariant |
| `TestIntensityStats` | `row_max`, `row_range`, `count_above_3`, `count_above_5` values |
| `TestSpatialFeatures` | All required spatial features (A–E); EC features with `--run-ec` |
| `TestIntegration` | Full pipeline shape `(N, 76)`, correct labels (0/1), no NaNs |

Tests will **fail** until you implement each TODO — that's expected. As you complete each part, more tests will pass. A passing required suite looks like:

```
============ 20 passed, 3 skipped in X.XXs ============
```

(The 3 skipped are extra credit tests for TODO F, hidden unless you pass `--run-ec`.)

## Hints

- `clean.py` is provided complete — use `from clean import clean_data`
- Look at numpy broadcasting: `axis=1, keepdims=True` is your friend
- For BFS, the grid is only 8x8 (64 cells) — keep it simple
- `np.where(condition)` returns indices where condition is True
- The lab tutorial's `scripts/features.py` has the complete reference implementation for all features
