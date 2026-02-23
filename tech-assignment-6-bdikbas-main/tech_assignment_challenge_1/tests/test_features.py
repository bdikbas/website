"""Tests for Challenge 1: Feature Engineering

Run required tests:
  uv run -m pytest tests/test_features.py -v

Run required + extra credit tests:
  uv run -m pytest tests/test_features.py -v --run-ec

Required tests cover Parts A, B, and spatial features A–E (spatial_gradient,
largest_blob, quadrant_var, center_vs_edge, row_profile_std, col_profile_std).
Extra credit tests cover TODO F (hot_centroid_r, hot_pixel_ratio).
Extra credit tests are skipped by default.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from features import (
    engineer_features,
    compute_spatial_features,
    _largest_connected_component,
)

# ============================================================
# Test fixtures: known input data
# ============================================================

# An "empty room" sample — all pixels near ambient (~23C), small variation
EMPTY_PIXELS = np.array([
    23.0, 23.1, 23.0, 22.9, 23.0, 23.1, 23.0, 23.0,
    23.1, 23.0, 23.0, 23.1, 23.0, 22.9, 23.0, 23.1,
    23.0, 23.0, 23.1, 23.0, 22.9, 23.0, 23.0, 23.0,
    22.9, 23.0, 23.0, 23.1, 23.0, 23.0, 23.0, 22.9,
    23.0, 23.0, 23.0, 23.0, 23.1, 23.0, 23.0, 23.0,
    23.0, 23.1, 23.0, 23.0, 23.0, 23.0, 22.9, 23.0,
    23.1, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.1,
    23.0, 23.0, 22.9, 23.0, 23.0, 23.1, 23.0, 23.0,
], dtype=np.float32)

# A "person present" sample — warm blob in center region
PRESENT_PIXELS = np.array([
    22.5, 22.6, 22.5, 22.7, 22.6, 22.5, 22.6, 22.5,
    22.6, 22.5, 23.0, 23.5, 23.8, 23.0, 22.5, 22.6,
    22.5, 23.0, 26.0, 27.5, 27.0, 26.5, 23.0, 22.5,
    22.7, 23.5, 27.0, 29.0, 28.5, 27.5, 23.5, 22.6,
    22.6, 23.0, 27.5, 28.5, 28.0, 27.0, 23.0, 22.5,
    22.5, 23.0, 26.0, 27.0, 26.5, 25.5, 23.0, 22.6,
    22.6, 22.5, 23.0, 23.5, 23.0, 23.0, 22.5, 22.5,
    22.5, 22.6, 22.5, 22.6, 22.5, 22.6, 22.5, 22.6,
], dtype=np.float32)

# A "person at edge" sample — warm region in top-left corner
EDGE_PIXELS = np.array([
    28.0, 27.5, 26.0, 23.0, 22.5, 22.5, 22.5, 22.5,
    27.0, 27.0, 25.5, 23.0, 22.5, 22.6, 22.5, 22.5,
    26.5, 25.5, 24.0, 22.5, 22.5, 22.5, 22.6, 22.5,
    23.0, 23.0, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5,
    22.5, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5,
    22.5, 22.5, 22.5, 22.5, 22.5, 22.6, 22.5, 22.5,
    22.5, 22.6, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5,
    22.5, 22.5, 22.5, 22.5, 22.5, 22.5, 22.6, 22.5,
], dtype=np.float32)


# ============================================================
# Part A: Normalization Tests
# ============================================================

class TestNormalization:
    def test_normalized_shape(self):
        """Normalized output should have 64 columns."""
        import pandas as pd
        pixel_cols = [f"pixel_{i}" for i in range(64)]
        df = pd.DataFrame([EMPTY_PIXELS], columns=pixel_cols)
        df["label"] = "empty"
        df["student_id"] = "test"
        X, y = engineer_features(df)
        # First 64 features are normalized pixels
        assert X.shape[1] == 76, f"Expected 76 features, got {X.shape[1]}"

    def test_normalized_mean_near_zero(self):
        """Normalized pixels should have mean approximately 0."""
        import pandas as pd
        pixel_cols = [f"pixel_{i}" for i in range(64)]
        df = pd.DataFrame([PRESENT_PIXELS], columns=pixel_cols)
        df["label"] = "present"
        df["student_id"] = "test"
        X, y = engineer_features(df)
        normalized = X[0, :64]
        assert abs(normalized.mean()) < 0.55, \
            f"Normalized mean should be near 0 (within 0.55), got {normalized.mean():.3f}"

    def test_normalization_ambient_invariant(self):
        """Same scene at different ambient temps should produce similar normalized values."""
        import pandas as pd
        pixel_cols = [f"pixel_{i}" for i in range(64)]

        # Same relative pattern, shifted by 5 degrees
        shifted = PRESENT_PIXELS + 5.0

        df1 = pd.DataFrame([PRESENT_PIXELS], columns=pixel_cols)
        df1["label"] = "present"
        df1["student_id"] = "test"
        X1, _ = engineer_features(df1)

        df2 = pd.DataFrame([shifted], columns=pixel_cols)
        df2["label"] = "present"
        df2["student_id"] = "test"
        X2, _ = engineer_features(df2)

        # Normalized pixels (first 64) should be very similar
        diff = np.abs(X1[0, :64] - X2[0, :64]).max()
        assert diff < 0.01, \
            f"Normalized pixels should be ambient-invariant, max diff = {diff:.4f}"


# ============================================================
# Part B: Intensity Statistics Tests
# ============================================================

class TestIntensityStats:
    def _get_features(self, pixels, label="present"):
        import pandas as pd
        pixel_cols = [f"pixel_{i}" for i in range(64)]
        df = pd.DataFrame([pixels], columns=pixel_cols)
        df["label"] = label
        df["student_id"] = "test"
        X, y = engineer_features(df)
        return X[0]

    def test_row_max(self):
        """Feature 64 should be the max pixel value."""
        feats = self._get_features(PRESENT_PIXELS)
        expected = PRESENT_PIXELS.max()
        assert abs(feats[64] - expected) < 0.01, \
            f"row_max: expected {expected}, got {feats[64]}"

    def test_row_range(self):
        """Feature 65 should be max - min pixel value."""
        feats = self._get_features(PRESENT_PIXELS)
        expected = PRESENT_PIXELS.max() - PRESENT_PIXELS.min()
        assert abs(feats[65] - expected) < 0.01, \
            f"row_range: expected {expected}, got {feats[65]}"

    def test_count_above_3(self):
        """Feature 66 should count pixels more than 3C above median."""
        feats = self._get_features(PRESENT_PIXELS)
        median = np.median(PRESENT_PIXELS)
        expected = (PRESENT_PIXELS > (median + 3.0)).sum()
        assert abs(feats[66] - expected) < 0.01, \
            f"count_above_3: expected {expected}, got {feats[66]}"

    def test_count_above_5(self):
        """Feature 67 should count pixels more than 5C above median."""
        feats = self._get_features(PRESENT_PIXELS)
        median = np.median(PRESENT_PIXELS)
        expected = (PRESENT_PIXELS > (median + 5.0)).sum()
        assert abs(feats[67] - expected) < 0.01, \
            f"count_above_5: expected {expected}, got {feats[67]}"

    def test_empty_room_low_counts(self):
        """Empty room should have 0 pixels above threshold."""
        feats = self._get_features(EMPTY_PIXELS, label="empty")
        assert feats[66] == 0.0, f"count_above_3 should be 0 for empty room, got {feats[66]}"
        assert feats[67] == 0.0, f"count_above_5 should be 0 for empty room, got {feats[67]}"


# ============================================================
# Part C: Spatial Features Tests
# Required: TODO A (spatial_gradient), TODO B (largest_blob),
#           TODO C (quadrant_var), TODO D (center_vs_edge),
#           TODO E (row_profile_std, col_profile_std)
# Extra Credit: TODO F (hot_centroid_r, hot_pixel_ratio)
# ============================================================

class TestSpatialFeatures:
    # ---- TODO A: spatial_gradient [REQUIRED] ----

    def test_spatial_gradient_present(self):
        """Person present should have higher spatial gradient than empty room."""
        present_feats = compute_spatial_features(PRESENT_PIXELS)
        empty_feats = compute_spatial_features(EMPTY_PIXELS)
        assert present_feats[0] > empty_feats[0], \
            f"Present gradient ({present_feats[0]:.3f}) should exceed empty ({empty_feats[0]:.3f})"

    # ---- TODO B: largest_blob [REQUIRED] ----

    def test_largest_blob_present(self):
        """Person present should have a large blob of hot pixels."""
        feats = compute_spatial_features(PRESENT_PIXELS)
        assert feats[1] >= 4, \
            f"Expected blob size >= 4 for present, got {feats[1]}"

    def test_largest_blob_empty(self):
        """Empty room should have no hot blob (blob size 0)."""
        feats = compute_spatial_features(EMPTY_PIXELS)
        assert feats[1] == 0, \
            f"Expected blob size 0 for empty room, got {feats[1]}"

    def test_largest_blob_bfs(self):
        """Test BFS directly on a known grid."""
        grid = np.zeros((8, 8), dtype=np.float32)
        # Create an L-shaped region above threshold
        grid[0, 0] = 10.0
        grid[0, 1] = 10.0
        grid[1, 0] = 10.0
        grid[2, 0] = 10.0
        # Isolated pixel
        grid[5, 5] = 10.0
        result = _largest_connected_component(grid, 5.0)
        assert result == 4, f"L-shaped region should have size 4, got {result}"

    # ---- TODO C: quadrant_var [REQUIRED] ----

    def test_quadrant_var_centered(self):
        """Person in center should have lower quadrant variance than person at edge."""
        center_feats = compute_spatial_features(PRESENT_PIXELS)
        edge_feats = compute_spatial_features(EDGE_PIXELS)
        # Edge person concentrates heat in one quadrant -> higher variance
        assert edge_feats[2] > center_feats[2], \
            f"Edge quadrant_var ({edge_feats[2]:.3f}) should exceed center ({center_feats[2]:.3f})"

    # ---- TODO D: center_vs_edge [REQUIRED] ----

    def test_center_vs_edge_present(self):
        """Person in center should have positive center_vs_edge."""
        feats = compute_spatial_features(PRESENT_PIXELS)
        assert feats[3] > 0, \
            f"center_vs_edge should be positive for centered person, got {feats[3]:.3f}"

    # ---- TODO E: row_profile_std / col_profile_std [REQUIRED] ----

    def test_row_profile_std_present(self):
        """Person present should have higher row_profile_std than empty room."""
        present_feats = compute_spatial_features(PRESENT_PIXELS)
        empty_feats = compute_spatial_features(EMPTY_PIXELS)
        assert present_feats[4] > empty_feats[4], \
            f"Present row_profile_std ({present_feats[4]:.3f}) should exceed empty ({empty_feats[4]:.3f})"

    def test_col_profile_std_present(self):
        """Person present should have higher col_profile_std than empty room."""
        present_feats = compute_spatial_features(PRESENT_PIXELS)
        empty_feats = compute_spatial_features(EMPTY_PIXELS)
        assert present_feats[5] > empty_feats[5], \
            f"Present col_profile_std ({present_feats[5]:.3f}) should exceed empty ({empty_feats[5]:.3f})"

    # ---- TODO F: hot_pixel_ratio [EXTRA CREDIT] ----

    @pytest.mark.extra_credit
    def test_hot_pixel_ratio_present(self):
        """Person present should have nonzero hot pixel ratio."""
        feats = compute_spatial_features(PRESENT_PIXELS)
        assert feats[7] > 0, \
            f"hot_pixel_ratio should be > 0 for present, got {feats[7]:.3f}"

    @pytest.mark.extra_credit
    def test_hot_pixel_ratio_empty(self):
        """Empty room should have zero hot pixel ratio."""
        feats = compute_spatial_features(EMPTY_PIXELS)
        assert feats[7] == 0.0, \
            f"hot_pixel_ratio should be 0 for empty room, got {feats[7]:.3f}"

    # ---- TODO F: hot_centroid_r [EXTRA CREDIT] ----

    @pytest.mark.extra_credit
    def test_hot_centroid_r_centered(self):
        """Person in center: centroid should be near grid center (small distance)."""
        center_feats = compute_spatial_features(PRESENT_PIXELS)
        edge_feats = compute_spatial_features(EDGE_PIXELS)
        # Centered person -> centroid near (3.5, 3.5) -> small distance
        # Edge person -> centroid far from center -> larger distance
        assert center_feats[6] < edge_feats[6], \
            f"Center hot_centroid_r ({center_feats[6]:.3f}) should be less than edge ({edge_feats[6]:.3f})"

    # ---- Always required ----

    def test_spatial_features_count(self):
        """compute_spatial_features should return exactly 8 values."""
        feats = compute_spatial_features(PRESENT_PIXELS)
        assert len(feats) == 8, f"Expected 8 spatial features, got {len(feats)}"


# ============================================================
# Integration Test
# ============================================================

class TestIntegration:
    def test_full_pipeline_shape(self):
        """Full pipeline should produce 76 features."""
        import pandas as pd
        pixel_cols = [f"pixel_{i}" for i in range(64)]
        rows = [EMPTY_PIXELS, PRESENT_PIXELS, EDGE_PIXELS]
        df = pd.DataFrame(rows, columns=pixel_cols)
        df["label"] = ["empty", "present", "present"]
        df["student_id"] = ["a", "b", "c"]
        X, y = engineer_features(df)
        assert X.shape == (3, 76), f"Expected shape (3, 76), got {X.shape}"
        assert y.shape == (3,), f"Expected y shape (3,), got {y.shape}"

    def test_labels(self):
        """Labels should be 0 for empty, 1 for present."""
        import pandas as pd
        pixel_cols = [f"pixel_{i}" for i in range(64)]
        rows = [EMPTY_PIXELS, PRESENT_PIXELS]
        df = pd.DataFrame(rows, columns=pixel_cols)
        df["label"] = ["empty", "present"]
        df["student_id"] = ["a", "b"]
        X, y = engineer_features(df)
        assert y[0] == 0.0, "empty label should map to 0"
        assert y[1] == 1.0, "present label should map to 1"

    def test_no_nans(self):
        """Output should contain no NaN values."""
        import pandas as pd
        pixel_cols = [f"pixel_{i}" for i in range(64)]
        rows = [EMPTY_PIXELS, PRESENT_PIXELS, EDGE_PIXELS]
        df = pd.DataFrame(rows, columns=pixel_cols)
        df["label"] = ["empty", "present", "present"]
        df["student_id"] = ["a", "b", "c"]
        X, y = engineer_features(df)
        assert not np.any(np.isnan(X)), "Feature matrix contains NaN values"
