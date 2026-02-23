"""Convert a pre-trained Keras model to TFLite INT8 and export C headers.

You are given:
  - pretrained_model.keras   (trained Keras model)
  - features.npz             (X array for representative dataset)
  - scaler.npz               (mean_ and scale_ arrays for StandardScaler)

You need to produce:
  - esp32/include/model_data.h    (model bytes as C array)
  - esp32/include/model_params.h  (scaler params as C arrays)

Usage:
  uv run scripts/export.py
"""

import os
import numpy as np
import tensorflow as tf


def convert_to_tflite(model, X_scaled):
    """Convert a Keras model to TFLite INT8 quantized format.

    Args:
        model: A trained Keras model.
        X_scaled: Scaled feature array for representative dataset calibration.

    Returns:
        bytes: The TFLite model as a byte string.
    """
    os.makedirs("model", exist_ok=True)

    # TODO 1: Create a representative dataset generator
    #   - Define a generator function that yields samples one at a time
    #   - Each yield should be a list containing one sample: [X_scaled[i:i+1].astype(np.float32)]
    #   - Use min(500, len(X_scaled)) samples
    def representative_dataset():
        n = min(500, len(X_scaled))
        for i in range(n):
            yield [X_scaled[i:i+1].astype(np.float32)]
    # Why INT8 quantization?
    # ----------------------
    # Neural networks are normally stored as 32-bit floats (~4 bytes per weight).
    # The ESP32-S3 has only ~380 KB of SRAM and runs at 240 MHz — not enough to
    # run a full float32 model in real time.
    #
    # INT8 quantization maps each float weight to an 8-bit integer (1 byte),
    # shrinking the model ~4x and making inference 2–4x faster on microcontrollers.
    #
    # The tradeoff: a tiny accuracy loss (usually < 1%) because we lose precision.
    # We calibrate the quantization ranges using the "representative dataset" above,
    # so the converter can choose the best float → int8 mapping for this data.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_path = "model/model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved {tflite_path} ({len(tflite_model)} bytes)")
    return tflite_model


def export_c_header(tflite_model, output_path="esp32/include/model_data.h"):
    """Write the TFLite model bytes as a C header file.

    The output should look like:
        #ifndef MODEL_DATA_H
        #define MODEL_DATA_H
        #include <cstdint>
        alignas(16) const unsigned char model_tflite[] = {
            0x1c, 0x00, 0x00, 0x00, ...
        };
        const unsigned int model_tflite_len = 1234;
        #endif

    Args:
        tflite_model: The TFLite model as bytes.
        output_path: Where to write the header file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # TODO 3: Write the model bytes as a C header file
    #   - Convert each byte to hex format: f"0x{byte:02x}"
    #   - Group into lines of 12 values each, indented with 4 spaces
    #   - Wrap in the header structure shown in the docstring above
    #   - Include: alignas(16) const unsigned char model_tflite[] = { ... };
    #   - Include: const unsigned int model_tflite_len = <length>;
    #   - Use #ifndef MODEL_DATA_H / #define MODEL_DATA_H / #endif guards
    hex_bytes = [f"0x{b:02x}" for b in tflite_model]
    lines = []
    for i in range(0, len(hex_bytes), 12):
        lines.append("    " + ", ".join(hex_bytes[i:i+12]) + ",")

    header = []
    header.append("#ifndef MODEL_DATA_H")
    header.append("#define MODEL_DATA_H")
    header.append("#include <cstdint>")
    header.append("alignas(16) const unsigned char model_tflite[] = {")
    header.extend(lines)
    header.append("};")
    header.append(f"const unsigned int model_tflite_len = {len(tflite_model)};")
    header.append("#endif")

    with open(output_path, "w") as f:
        f.write("\n".join(header) + "\n")


def export_scaler_params(mean, scale, output_path="esp32/include/model_params.h"):
    """Write StandardScaler parameters as a C header file.

    The output should look like:
        #ifndef MODEL_PARAMS_H
        #define MODEL_PARAMS_H
        const int N_FEATURES = 76;
        const float SCALER_MEAN[76] = { ... };
        const float SCALER_SCALE[76] = { ... };
        #endif

    Args:
        mean: Array of scaler means (shape: n_features).
        scale: Array of scaler scales (shape: n_features).
        output_path: Where to write the header file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # TODO 4: Write scaler parameters as a C header file
    #   - Get n_features from len(mean)
    #   - Format each value as f"{v:.6f}f"
    #   - Write N_FEATURES, SCALER_MEAN array, and SCALER_SCALE array
    #   - Use #ifndef MODEL_PARAMS_H / #define MODEL_PARAMS_H / #endif guards
    n_features = len(mean)

    mean_vals = ", ".join([f"{v:.6f}f" for v in mean])
    scale_vals = ", ".join([f"{v:.6f}f" for v in scale])

    header = []
    header.append("#ifndef MODEL_PARAMS_H")
    header.append("#define MODEL_PARAMS_H")
    header.append(f"const int N_FEATURES = {n_features};")
    header.append(f"const float SCALER_MEAN[{n_features}] = {{ {mean_vals} }};")
    header.append(f"const float SCALER_SCALE[{n_features}] = {{ {scale_vals} }};")
    header.append("#endif")

    with open(output_path, "w") as f:
        f.write("\n".join(header) + "\n")


if __name__ == "__main__":
    # Load the pre-trained model
    model = tf.keras.models.load_model("pretrained_model.keras")

    # Load pre-computed features for representative dataset
    data = np.load("features.npz")
    X_scaled = data["X_scaled"]

    # Load pre-fit scaler parameters
    scaler_data = np.load("scaler.npz")
    mean = scaler_data["mean"]
    scale = scaler_data["scale"]

    # Run the export pipeline
    tflite_model = convert_to_tflite(model, X_scaled)
    export_c_header(tflite_model)
    export_scaler_params(mean, scale)

    print("\nDone! Check esp32/include/ for model_data.h and model_params.h")
