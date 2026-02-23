# Lab Challenge: TFLite Conversion + ESP32 Deployment

You are given a **pre-trained Keras model** and **pre-computed features**. Your job is to convert the model to TFLite INT8, export it as C headers, and wire up the ESP32 inference pipeline.

**Files you have:**
- `pretrained_model.keras` — trained model (no training needed)
- `features.npz` — pre-computed scaled features for the representative dataset
- `scaler.npz` — StandardScaler params (mean, scale)

## Part 1 — Python: TFLite Conversion

Open `scripts/export.py` and fill in the TODOs:

1. **TODO 1** — Representative dataset generator
2. **TODO 2** — TFLite converter config *(provided — read the comments to understand why INT8 quantization)*
3. **TODO 3** — C header export for model bytes
4. **TODO 4** — C header export for scaler params

> **Why is TODO 2 given?** INT8 quantization configuration is the same boilerplate in every TFLite project. The important thing is understanding *why* it's done — the comments explain the tradeoff between model size, inference speed, and accuracy. Read them.

From inside the folder `lab_challenge` run:
```bash
uv run scripts/export.py
```

If it works you should see something like:
```
Saved model/model.tflite (XXXX bytes)
Saved esp32/include/model_data.h
Saved esp32/include/model_params.h
```

Check that those files exist in `esp32/include/`.

## Part 2 — ESP32: Inference Pipeline

Open `esp32/src/main.cpp` and do the 2 TODOs:

- **TODO 1** — `setupModel()` — load TFLite model, create interpreter, allocate tensors
- **TODO 2** — `runInference()` — quantize float→int8, run model, dequantize int8→float

Sensor reading and `computeFeatures()` are already done.

**INT8 quantization:** The model uses 8-bit integers instead of floats. So you float features → int8 before feeding in, and the model outputs int8 → you convert back to float. The scale and zero_point params live in each tensor:

```
quantize:   int8_value = round(float_value / scale) + zero_point
dequantize: float_value = (int8_value - zero_point) * scale
```

## Part 3 — Flash and Verify

1. Copy `esp32/env.example` to `esp32/.env` and fill in your WiFi credentials
2. Upload to ESP32 using PlatformIO

> **Note on warnings:** Building the C++ prints many warning messages from the ESP32 toolchain and TFLite library. This is **completely normal**. Only lines with `error:` matter. A build ending with `[SUCCESS]` is a good build.

You should see output like:
```
[TFLite] Input: 76 dims, type=1
[TFLite] Arena used: 3456 bytes
[OK] Model loaded, starting inference loop
[PRESENT] conf=0.873 | min=22.5C max=31.2C
[EMPTY  ] conf=0.041 | min=22.1C max=23.4C
```

## The Key Insight

Your `export.py` (Part 1) generates the files that your `main.cpp` (Part 2) `#include`s.
The Python → C header → ESP32 pipeline is the same pattern you'll use in the tech assignments.

## Hints

- Look at the lab tutorial's `scripts/export.py` for the complete export implementation
- Look at the lab tutorial's `esp32/src/main.cpp` for the complete `setupModel()` and `runInference()`
- The TFLite Micro API uses `tflite::GetModel()`, `tflite::AllOpsResolver`, and `tflite::MicroInterpreter`

## **Deliverables**

Submit a video to the gradescope assignment **Lab 7** of your model working on your ESP. 
 - The video must clearly show the ESP 32 and the connected AMG8833 sensor connected to it
 - The video must clearly show you pointing the sensor at a person (can be yourself) and the serial output showing a "PRESENT" detection. 
 - The video must clearly show you pointing the sensor at something other than a person, and the resulting serial output showing "EMPTY".
 - Video should be 30-60 seconds.