# Tech Assignment Challenge 2 — Train, Convert, and Deploy

Build the complete ML pipeline: train a model on your engineered features, convert it to
TFLite INT8, export C headers, and deploy to an ESP32 for real-time inference.

## Prerequisites

- Your completed `features.py` from Challenge 1
- `thermal_dataset.csv` (provided)

## Setup

Copy your `features.py` from Challenge 1:
```bash
cp ../tech_assignment_challenge_1/scripts/features.py scripts/features.py
```

## Part A — Train the Model (`scripts/train.py`)

Complete the 4 TODOs in `scripts/train.py`:

| TODO | What to implement | What you'll learn |
|---|---|---|
| **TODO A1** | Fit StandardScaler on train set, transform both sets | Feature scaling for neural networks |
| **TODO A2** | Build Keras Sequential model | Neural network architecture design |
| **TODO A3** | Train with EarlyStopping + ReduceLROnPlateau | Training best practices, callbacks |
| **TODO A4** | Evaluate and print classification report | Model evaluation metrics |

Run training:
```bash
uv run scripts/train.py
```

This saves `trained_model.keras`, `scaler.npz`, and `features.npz` for the export step.

### Model Architecture

**Required:**
```
Input(76) -> Dense(32, relu, L2) -> Dense(16, relu, L2) -> Dense(1, sigmoid)
```
- Regularization: `keras.regularizers.l2(0.005)`
- Optimizer: `adam`
- Loss: `binary_crossentropy`
- Callbacks: EarlyStopping (patience=20) + ReduceLROnPlateau (factor=0.5, patience=10)

**Extra Credit:** Extend the model with L1_L2 regularization and Dropout:
```
Input(76) -> Dense(32, relu, L1_L2) -> Dropout(0.3) -> Dense(16, relu, L1_L2) -> Dropout(0.3) -> Dense(1, sigmoid)
```
- **L1 regularization** drives some weights to exactly zero (sparse model, built-in feature selection)
- **L2 regularization** penalizes large weights (spreads weight values more evenly)
- **Dropout** randomly disables neurons during training (forces redundant representations, reduces co-adaptation)

## Part B — Export to TFLite (`scripts/export.py`)

Complete the TODOs in `scripts/export.py` (same pattern as the lab challenge):

| TODO | What to implement |
|---|---|
| **TODO 1** | Representative dataset generator |
| **TODO 2** | TFLite INT8 converter configuration *(provided — read the comments)* |
| **TODO 3** | Export model bytes as C header |
| **TODO 4** | Export scaler params as C header |

Run export:
```bash
uv run scripts/export.py
```

This generates `esp32/include/model_data.h` and `esp32/include/model_params.h`.

## Part C — Deploy to ESP32

The `esp32/` project is provided complete. Your generated headers are the only missing piece.

1. Copy `esp32/env.example` to `esp32/.env` and fill in your WiFi credentials
2. Flash the ESP32 using PlatformIO
3. Verify predictions make sense — `[PRESENT]` when pointing at yourself, `[EMPTY]` for empty room

**Extra Credit:** The provided `esp32/src/main.cpp` deliberately zeroes out two features after scaling, preventing them from contributing to inference even though they are computed. If you implemented the extra credit features in Challenge 1, edit `tech_assignment_challenge_2/esp32/src/main.cpp` to allow the features to be computed correctly.
> **Note on warnings:** Building the C++ code will print many warning messages from the ESP32 toolchain and the TFLite library. This is **completely normal**. Ignore anything labeled `warning:` — only `error:` lines are actual problems. A build ending with `[SUCCESS]` means everything is fine.

## Running Tests

Tests live in `tests/test_export.py`. They validate your exported artifacts — **run `train.py` and `export.py` first** to generate the files the tests check.

```bash
uv run -m pytest tests/test_export.py -v
```

What each group tests:

| Test class | What it checks |
|---|---|
| `TestTFLiteModel` | `model.tflite` exists, is non-empty, < 50 KB, input/output are INT8, input shape is 76 |
| `TestModelDataHeader` | `model_data.h` has include guards, `model_tflite[]` array, `alignas(16)`, declared length matches actual hex byte count |
| `TestModelParamsHeader` | `model_params.h` has include guards, `N_FEATURES=76`, `SCALER_MEAN` and `SCALER_SCALE` arrays with 152 float values |

Tests will be **skipped** (not failed) if the output files don't exist yet. Once you've run `export.py`, they will run fully. A passing suite looks like:

```
============ 12 passed in X.XXs ============
```

## Hints

- The lab tutorial's `scripts/train.py` and `scripts/export.py` are your reference
- If you completed the lab challenge export, reuse that code for Part B
- `StandardScaler.fit_transform()` fits AND transforms in one call (use on train only!)
- `StandardScaler.transform()` transforms without fitting (use on validation set)
- The `GroupKFold` split is provided — focus on the scaler, model, and training code

## Full Pipeline

When everything works, the complete flow is:

```
thermal_dataset.csv
    -> clean.py (clean)
    -> features.py (76 features)    [YOUR Challenge 1 code]
    -> train.py (Keras model)       [YOUR Challenge 2A code]
    -> export.py (TFLite + headers) [YOUR Challenge 2B code]
    -> ESP32 (real-time inference)
```
You now own every piece of this pipeline.

## **Deliverables**

Include a video link at the very bottom of this file
 - The video must clearly show the ESP 32 and the connected AMG8833 sensor connected to it
 - The video must clearly show you pointing the sensor at a person (can be yourself) and the serial output showing a "PRESENT" detection. 
 - The video must clearly show you pointing the sensor at something other than a person, and the resulting serial output showing "EMPTY".
 - The video **MUST** show you point at areas without a person and areas with a person using the sensor, and the serial output must respond accordingly.
 - Video should be 30-90 seconds.

## VIDEO LINK
[VIDEO LINK GOES BELOW]
