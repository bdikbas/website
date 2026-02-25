"""Challenge 2, Part A: Train a model on your engineered features.

You are given:
  - Your features.py from Challenge 1 (import it)
  - clean.py (provided)
  - thermal_dataset.csv

You need to:
  TODO A1: Fit a StandardScaler on the training set, transform train + val
  TODO A2: Build a Keras Sequential model:
           Required:     Input(76) -> Dense(32, relu, L2) -> Dense(16, relu, L2) -> Dense(1, sigmoid)
           Extra Credit: Input(76) -> Dense(32, relu, L1_L2) -> Dropout(0.3) -> Dense(16, relu, L1_L2) -> Dropout(0.3) -> Dense(1, sigmoid)
  TODO A3: Train with EarlyStopping and ReduceLROnPlateau callbacks
  TODO A4: Print held-out accuracy and classification report

Usage:
  uv run scripts/train.py
"""

import sys
import os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(__file__))
from clean import clean_data
from features import engineer_features


def build_model(input_dim=76):
    """Build the Keras Sequential model.

    Required architecture:
        Input(input_dim)
        -> Dense(32, relu) with L2 regularization (l2=0.005)
        -> Dense(16, relu) with L2 regularization
        -> Dense(1, sigmoid)

    Extra Credit architecture:
        Input(input_dim)
        -> Dense(32, relu) with L1_L2 regularization (l1=0.005, l2=0.005)
        -> Dropout(0.3)
        -> Dense(16, relu) with L1_L2 regularization
        -> Dropout(0.3)
        -> Dense(1, sigmoid)

    Compile with: optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]

    Args:
        input_dim: Number of input features (default: 76).

    Returns:
        A compiled Keras model.
    """
    # TODO A2: Build the model architecture
    #   Required:
    #   - Create a regularizer: keras.regularizers.l2(0.005)
    #   - Build a Sequential model with the layers described above
    #   - Compile with adam optimizer and binary_crossentropy loss
    #
    #   Extra Credit:
    #   - Use keras.regularizers.l1_l2(l1=0.005, l2=0.005) instead of l2
    #   - Add Dropout(0.3) after each hidden Dense layer
    reg = keras.regularizers.l2(0.005)

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_final_model(X, y, groups, n_splits=5):
    """Train a final model using GroupKFold for the train/val split.

    Uses the last fold split: trains on all-but-one fold, validates on the held-out fold.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Labels, shape (n_samples,).
        groups: Group labels for GroupKFold (student_id).
        n_splits: Number of folds.

    Returns:
        tuple: (model, scaler, X_all_scaled, accuracy)
            - model: Trained Keras model
            - scaler: StandardScaler fit on ALL data (for deployment)
            - X_all_scaled: All features scaled (for representative dataset)
            - accuracy: Held-out accuracy
    """
    # Split using GroupKFold (provided — this is not the learning goal)
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))
    train_idx, val_idx = splits[-1]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # TODO A1: Fit a StandardScaler on X_train and transform both sets
    #   - Create a StandardScaler
    #   - Fit on X_train and transform X_train -> X_train_scaled
    #   - Transform X_val -> X_val_scaled (do NOT fit on val!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # TODO A3: Train the model with callbacks
    #   - Build the model using build_model(X.shape[1])
    #   - Create callbacks:
    #       EarlyStopping: monitor="val_accuracy", patience=20, restore_best_weights=True
    #       ReduceLROnPlateau: monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
    #   - Call model.fit() with:
    #       X_train_scaled, y_train
    #       validation_data=(X_val_scaled, y_val)
    #       epochs=200, batch_size=32
    #       callbacks=callbacks, verbose=1
    model = build_model(X.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # TODO A4: Evaluate and print results
    #   - Predict on X_val_scaled (model.predict)
    #   - Threshold at 0.5 to get binary predictions
    #   - Compute accuracy with accuracy_score
    #   - Print classification_report with target_names=["empty", "present"]
    y_prob = model.predict(X_val_scaled, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    acc = accuracy_score(y_val, y_pred)
    print("\nClassification report (held-out fold):")
    print(classification_report(
        y_val, y_pred,
        target_names=["empty", "present"],
        digits=4
    ))

    # Fit a final scaler on ALL data (for deployment)
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X)

    return model, final_scaler, X_all_scaled, acc


if __name__ == "__main__":
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "thermal_dataset.csv"))

    df_clean = clean_data(csv_path)
    X, y = engineer_features(df_clean)

    # Need groups for GroupKFold — extract from DataFrame
    df_clean2 = clean_data(csv_path)
    groups = df_clean2["student_id"].values

    model, scaler, X_scaled, acc = train_final_model(X, y, groups)

    if model is not None:
        model.summary()
        print(f"\nHeld-out accuracy: {acc:.1%}")

        # Save for use with export.py
        model.save("trained_model.keras")
        np.savez("scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
        np.savez("features.npz", X_scaled=X_scaled)
        print("Saved trained_model.keras, scaler.npz, features.npz")
    else:
        print("Model not implemented yet — complete the TODOs!")
