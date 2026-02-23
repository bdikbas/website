"""Train model with GroupKFold + L2 regularization

Uses GroupKFold (k=5) with student_id as group, and a Dense network
with L2 regularization.

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


def build_model(input_dim):
    reg = keras.regularizers.l2(0.005)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_with_group_kfold(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    fold_accs = []

    print(f"GroupKFold (k={n_splits}):")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = build_model(X.shape[1])
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=50, batch_size=32, verbose=0)

        y_pred = (model.predict(X_val, verbose=0).flatten() > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        fold_accs.append(acc)

        val_students = np.unique(groups[val_idx])
        print(f"  Fold {fold+1}: {acc:.1%} ({len(val_students)} students held out)")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"  CV mean: {mean_acc:.1%} +/- {std_acc:.1%}")
    return fold_accs, mean_acc, std_acc


def train_final_model(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))
    train_idx, val_idx = splits[-1]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training final model...")
    model = build_model(X.shape[1])
    model.summary()

    model.fit(X_train_scaled, y_train,
              validation_data=(X_val_scaled, y_val),
              epochs=50, batch_size=32, verbose=1)

    y_pred = (model.predict(X_val_scaled, verbose=0).flatten() > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    print(f"Held-out accuracy: {acc:.1%}")
    print(classification_report(y_val, y_pred, target_names=["empty", "present"]))

    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X)

    return model, final_scaler, X_all_scaled, acc


if __name__ == "__main__":
    df_clean = clean_data()
    X, y, groups = engineer_features(df_clean)

    fold_accs, mean_acc, std_acc = train_with_group_kfold(X, y, groups)
    model, scaler, X_scaled, acc = train_final_model(X, y, groups)

    print(f"Done. CV: {mean_acc:.1%} +/- {std_acc:.1%}, held-out: {acc:.1%}")
