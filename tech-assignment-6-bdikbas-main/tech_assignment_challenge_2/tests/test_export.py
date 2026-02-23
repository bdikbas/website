"""Tests for Challenge 2: Model Export

Run with:
  uv run -m pytest tests/test_export.py -v

These tests validate that:
  1. The exported TFLite model is properly INT8 quantized
  2. The generated C headers are well-formed

Note: You must run train.py and export.py first to generate the artifacts.
"""

import os
import re
import numpy as np
import pytest

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")


class TestTFLiteModel:
    """Validate the exported TFLite model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = os.path.join(BASE_DIR, "model", "model.tflite")
        if not os.path.exists(self.model_path):
            pytest.skip("model/model.tflite not found — run export.py first")

    def test_model_exists(self):
        """TFLite model file should exist."""
        assert os.path.exists(self.model_path)

    def test_model_not_empty(self):
        """TFLite model should have non-zero size."""
        size = os.path.getsize(self.model_path)
        assert size > 0, "model.tflite is empty"

    def test_model_reasonable_size(self):
        """TFLite INT8 model should be small (< 50KB for this architecture)."""
        size = os.path.getsize(self.model_path)
        assert size < 50_000, f"Model is {size} bytes — too large for INT8 quantized"

    def test_model_is_int8(self):
        """Model input/output should be INT8 quantized."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        assert input_details[0]["dtype"] == np.int8, \
            f"Input dtype should be int8, got {input_details[0]['dtype']}"
        assert output_details[0]["dtype"] == np.int8, \
            f"Output dtype should be int8, got {output_details[0]['dtype']}"

    def test_model_input_shape(self):
        """Model should accept 76 features as input."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_shape = input_details[0]["shape"]
        assert input_shape[-1] == 76, \
            f"Expected 76 input features, got {input_shape[-1]}"


class TestModelDataHeader:
    """Validate the generated model_data.h."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.header_path = os.path.join(BASE_DIR, "esp32", "include", "model_data.h")
        if not os.path.exists(self.header_path):
            pytest.skip("model_data.h not found — run export.py first")
        with open(self.header_path, "r") as f:
            self.content = f.read()

    def test_include_guard(self):
        """Header should have #ifndef/#define/#endif guards."""
        assert "#ifndef MODEL_DATA_H" in self.content
        assert "#define MODEL_DATA_H" in self.content
        assert "#endif" in self.content

    def test_has_model_array(self):
        """Header should contain the model_tflite array declaration."""
        assert "model_tflite[]" in self.content
        assert "alignas(16)" in self.content

    def test_has_model_length(self):
        """Header should contain model_tflite_len."""
        assert "model_tflite_len" in self.content

    def test_hex_format(self):
        """Array values should be in 0xNN hex format."""
        hex_pattern = r"0x[0-9a-f]{2}"
        matches = re.findall(hex_pattern, self.content)
        assert len(matches) > 100, \
            f"Expected many hex values, found {len(matches)}"

    def test_length_matches_bytes(self):
        """model_tflite_len should match the number of hex bytes."""
        hex_pattern = r"0x[0-9a-f]{2}"
        hex_count = len(re.findall(hex_pattern, self.content))

        len_match = re.search(r"model_tflite_len\s*=\s*(\d+)", self.content)
        assert len_match is not None, "Could not find model_tflite_len value"
        declared_len = int(len_match.group(1))

        assert hex_count == declared_len, \
            f"Declared length {declared_len} doesn't match hex byte count {hex_count}"


class TestModelParamsHeader:
    """Validate the generated model_params.h."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.header_path = os.path.join(BASE_DIR, "esp32", "include", "model_params.h")
        if not os.path.exists(self.header_path):
            pytest.skip("model_params.h not found — run export.py first")
        with open(self.header_path, "r") as f:
            self.content = f.read()

    def test_include_guard(self):
        """Header should have #ifndef/#define/#endif guards."""
        assert "#ifndef MODEL_PARAMS_H" in self.content
        assert "#define MODEL_PARAMS_H" in self.content
        assert "#endif" in self.content

    def test_has_n_features(self):
        """Header should define N_FEATURES = 76."""
        match = re.search(r"N_FEATURES\s*=\s*(\d+)", self.content)
        assert match is not None, "N_FEATURES not found"
        assert int(match.group(1)) == 76, \
            f"N_FEATURES should be 76, got {match.group(1)}"

    def test_has_scaler_mean(self):
        """Header should contain SCALER_MEAN array."""
        assert "SCALER_MEAN" in self.content

    def test_has_scaler_scale(self):
        """Header should contain SCALER_SCALE array."""
        assert "SCALER_SCALE" in self.content

    def test_float_format(self):
        """Scaler values should be in float format (e.g., 1.234567f)."""
        float_pattern = r"-?\d+\.\d+f"
        matches = re.findall(float_pattern, self.content)
        # Should have 76 mean values + 76 scale values = 152 floats
        assert len(matches) >= 152, \
            f"Expected at least 152 float values, found {len(matches)}"
