"""Export Model & Parameters

Converts trained Keras model to TFLite INT8 quantized format, and exports:
  - model/model.tflite
  - esp32/include/model_data.h
  - esp32/include/model_params.h

Usage:
  uv run scripts/export.py
"""

import os
import numpy as np
import tensorflow as tf


def convert_to_tflite(model, X_scaled):
    os.makedirs("model", exist_ok=True)

    def representative_dataset():
        for i in range(min(500, len(X_scaled))):
            yield [X_scaled[i:i+1].astype(np.float32)]

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    hex_vals = [f"0x{b:02x}" for b in tflite_model]
    lines = []
    for i in range(0, len(hex_vals), 12):
        lines.append("    " + ", ".join(hex_vals[i:i+12]))
    hex_body = ",\n".join(lines)

    header = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <cstdint>

alignas(16) const unsigned char model_tflite[] = {{
{hex_body}
}};

const unsigned int model_tflite_len = {len(tflite_model)};

#endif
"""
    with open(output_path, "w") as f:
        f.write(header)
    print(f"Saved {output_path}")


def export_scaler_params(scaler, output_path="esp32/include/model_params.h"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_features = len(scaler.mean_)
    mean_str = ", ".join(f"{v:.6f}f" for v in scaler.mean_)
    scale_str = ", ".join(f"{v:.6f}f" for v in scaler.scale_)

    header = f"""#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

// StandardScaler params: {n_features} features
const int N_FEATURES = {n_features};

const float SCALER_MEAN[{n_features}] = {{
    {mean_str}
}};

const float SCALER_SCALE[{n_features}] = {{
    {scale_str}
}};

#endif
"""
    with open(output_path, "w") as f:
        f.write(header)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from clean import clean_data
    from features import engineer_features
    from train import train_final_model

    df_clean = clean_data()
    X, y, groups = engineer_features(df_clean)
    model, scaler, X_scaled, acc = train_final_model(X, y, groups)

    tflite_model = convert_to_tflite(model, X_scaled)
    export_c_header(tflite_model)
    export_scaler_params(scaler)
