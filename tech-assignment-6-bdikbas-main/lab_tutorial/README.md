# Lab Tutorial — Thermal Presence Detection

Run the pipeline end-to-end and see how it works — raw sensor data to a model on the ESP32.

**No code changes needed.** Just run it and follow along.

## What it does

An AMG8833 thermal sensor captures an 8x8 grid of temperature readings. A TinyML model on an
ESP32 classifies each frame as "present" (someone is in view) or "empty" (no one). The pipeline:

1. **Clean** — remove sensor glitches, flat readings, ambiguous labels
2. **Features** — 64 raw pixels + 1 BFS blob feature = 65 total
3. **Train** — Dense net with L2 regularization, GroupKFold cross-validation
4. **Export** — TFLite INT8 + C header files
5. **Deploy** — ESP32 real-time inference

## Prerequisites

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- PlatformIO for ESP32
- ESP32-S3 + AMG8833 sensor

## Run the ML pipeline

```bash
uv run main.py
```

This runs the full pipeline. Watch the output — you'll see:
- **Cleaning**: how many rows are kept vs discarded, and why
- **Features**: 65 features (64 raw pixels + largest connected blob via BFS)
- **Training**: GroupKFold cross-validation accuracy per fold, then a final held-out model
- **Export**: TFLite model size, C header files generated

Output files:
- `model/model.tflite` — the quantized model
- `esp32/include/model_data.h` — model bytes as a C array
- `esp32/include/model_params.h` — StandardScaler parameters as C arrays

## Look at the scripts

Each step lives in `scripts/`:
- `clean.py` — flags glitches, flat readings, ambiguous labels
- `features.py` — 64 raw pixels + BFS largest blob size
- `train.py` — StandardScaler + Dense(32) → Dense(16) → Dense(1) with L2 regularization, GroupKFold
- `export.py` — TFLite INT8 with representative dataset, C header generation

## Flash the ESP32

1. Copy `esp32/env.example` to `esp32/.env` and fill in your WiFi credentials
2. Flash with PlatformIO

> **Note on warnings:** Building the C++ will print many warning messages from the ESP32 toolchain and TFLite library. This is **normal**. Only lines containing `error:` are actual problems. As long as the build ends with `[SUCCESS]`, everything is fine.

## Live inference

Serial monitor should show something like:

```
[TFLite] Input: 65 dims, type=1
[TFLite] Arena used: 3456 bytes
[OK] Model loaded, starting inference loop
[PRESENT] conf=0.873 | min=22.5C max=31.2C
[PRESENT] conf=0.912 | min=22.3C max=30.8C
[EMPTY  ] conf=0.041 | min=22.1C max=23.4C
```

- Point the sensor at yourself — you should see `[PRESENT]` with high confidence
- Point it at an empty room — you should see `[EMPTY]` with low confidence
- Notice how the confidence changes as you move in and out of the sensor's field of view

## Things to understand

- **BFS blob detection** — a person creates a contiguous warm region; the largest connected component of "hot" pixels captures this
- **GroupKFold** — splits by student_id so the model is tested on students it's never seen, giving a realistic accuracy estimate
- **L2 regularization** — penalizes large weights to prevent overfitting on a small dataset
- **INT8 quantization** — 8-bit integers instead of 32-bit floats, ~4x smaller
- **Representative dataset** — TFLite needs example inputs during conversion to calibrate quantization ranges
- **Python -> C header** — Python spits out `.h` files that C++ `#include`s. That's the bridge between training and deployment.

## What this tutorial doesn't do (but the challenges will)

This tutorial uses raw pixels + one spatial feature and a simple L2-regularized model. In the challenges, you'll improve it with:
- **Ambient normalization** — per-sample median/std normalization so different room temps don't matter
- **Intensity statistics** — count of hot pixels, temperature range
- **More spatial features** — gradients, quadrant variance, centroid distance, profile std (some extra credit)
- **L1 regularization + Dropout** — additional regularization techniques (introduced in Challenge 1, used in Challenge 2)
- **Training callbacks** — EarlyStopping, ReduceLROnPlateau for smarter training
