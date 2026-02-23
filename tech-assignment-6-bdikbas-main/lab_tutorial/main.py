"""Thermal Presence Detection — Full Pipeline

Orchestrates: clean -> features -> GroupKFold train -> export

Usage:
  uv run main.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from clean import clean_data
from features import engineer_features
from train import train_with_group_kfold, train_final_model
from export import convert_to_tflite, export_c_header, export_scaler_params


def main():
    print("Thermal Presence Detection — Clean, Train & Export\n")

    df_clean = clean_data()
    X, y, groups = engineer_features(df_clean)

    fold_accs, mean_acc, std_acc = train_with_group_kfold(X, y, groups)
    model, scaler, X_scaled, acc = train_final_model(X, y, groups)

    tflite_model = convert_to_tflite(model, X_scaled)
    export_c_header(tflite_model)
    export_scaler_params(scaler)

    print(f"\nDone. {len(tflite_model)} bytes, CV {mean_acc:.1%} +/- {std_acc:.1%}, held-out {acc:.1%}")


if __name__ == "__main__":
    main()
