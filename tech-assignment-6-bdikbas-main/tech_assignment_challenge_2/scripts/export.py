"""Challenge 2, Part B: Convert your trained model to TFLite INT8 and export C headers.

Same as lab challenge export, but you're converting your own model.
Inputs: trained_model.keras, features.npz, scaler.npz
Outputs: esp32/include/model_data.h, model_params.h

Run: uv run scripts/export.py
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
    #   - Each yield should be: [X_scaled[i:i+1].astype(np.float32)]
    #   - Use min(500, len(X_scaled)) samples
    def representative_dataset():
        n = min(500, len(X_scaled))
        for i in range(n):
            yield [X_scaled[i:i+1].astype(np.float32)]

    # TODO 2: Configure the TFLite converter for INT8 quantization
    #   - Create converter: tf.lite.TFLiteConverter.from_keras_model(model)
    #   - Set converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #   - Set converter.representative_dataset = representative_dataset
    #   - Set converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #   - Set converter.inference_input_type = tf.int8
    #   - Set converter.inference_output_type = tf.int8
    #   - Call converter.convert()
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

    Output format:
        #ifndef MODEL_DATA_H
        #define MODEL_DATA_H
        #include <cstdint>
        alignas(16) const unsigned char model_tflite[] = {
            0x1c, 0x00, 0x00, ...
        };
        const unsigned int model_tflite_len = <length>;
        #endif
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # TODO 3: Write the model bytes as a C header file
    #   - Convert each byte to hex format: f"0x{byte:02x}"
    #   - Group into lines of 12 values, indented with 4 spaces
    #   - Wrap in the header structure shown above
    hex_bytes = [f"0x{b:02x}" for b in tflite_model]

    lines = []
    for i in range(0, len(hex_bytes), 12):
        chunk = ", ".join(hex_bytes[i:i+12])
        lines.append(f"    {chunk},")

    content = "\n".join([
        "#ifndef MODEL_DATA_H",
        "#define MODEL_DATA_H",
        "#include <cstdint>",
        "",
        "alignas(16) const unsigned char model_tflite[] = {",
        *lines,
        "};",
        f"const unsigned int model_tflite_len = {len(tflite_model)};",
        "",
        "#endif",
        "",
    ])

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Wrote {output_path}")


def export_scaler_params(mean, scale, output_path="esp32/include/model_params.h"):
    """Write StandardScaler parameters as a C header file.

    Output format:
        #ifndef MODEL_PARAMS_H
        #define MODEL_PARAMS_H
        const int N_FEATURES = <n>;
        const float SCALER_MEAN[<n>] = { ... };
        const float SCALER_SCALE[<n>] = { ... };
        #endif
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # TODO 4: Write scaler parameters as a C header file
    #   - n_features = len(mean)
    #   - Format values as f"{v:.6f}f"
    #   - Write N_FEATURES, SCALER_MEAN array, SCALER_SCALE array
    n_features = len(mean)

    mean_str = ", ".join([f"{v:.6f}f" for v in mean])
    scale_str = ", ".join([f"{v:.6f}f" for v in scale])

    content = "\n".join([
        "#ifndef MODEL_PARAMS_H",
        "#define MODEL_PARAMS_H",
        "",
        f"const int N_FEATURES = {n_features};",
        f"const float SCALER_MEAN[{n_features}] = {{ {mean_str} }};",
        f"const float SCALER_SCALE[{n_features}] = {{ {scale_str} }};",
        "",
        "#endif",
        "",
    ])

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    # Load your trained model
    model = tf.keras.models.load_model("trained_model.keras")

    # Load pre-computed features and scaler
    data = np.load("features.npz")
    X_scaled = data["X_scaled"]

    scaler_data = np.load("scaler.npz")
    mean = scaler_data["mean"]
    scale = scaler_data["scale"]

    # Run the export pipeline
    tflite_model = convert_to_tflite(model, X_scaled)
    export_c_header(tflite_model)
    export_scaler_params(mean, scale)

    print("\nDone! Check esp32/include/ for model_data.h and model_params.h")
