[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Me5moLg0)
# Lab 7 and Tech Assignment 6

This week we will focus on using the data that was collected last week.

> **⚠️IMPORTANT⚠️**:
> Expect to spend more time on this tech assignment than last week! It **will** be more difficult

---

## Overview

The main focus for this week is training and deploying machine learning models on our ESP32. You will learn how we can quantize models, and implement data processing/preprocessing pipelines.

The project uses an AMG8833 thermal sensor we have been using to detect whether a person is present or not. You'll work through the full pipeline: cleaning raw sensor data, engineering features, training a neural network, converting it to TFLite INT8, and deploying it to an ESP32 for real-time inference.

## Structure

### Lab

| Folder | What it is |
|---|---|
| `lab_tutorial/` | Complete working pipeline — run it, read the code, understand the flow. Nothing to modify. |
| `lab_challenge/` | Given a pre-trained model, implement TFLite INT8 export and ESP32 inference. Quantization code is provided — focus on understanding it. |

### Tech Assignment

| Folder | What it is |
|---|---|
| `tech_assignment_challenge_1/` | Implement feature engineering: ambient normalization, intensity stats, and 6 required spatial features. 2 additional spatial features are **extra credit**. |
| `tech_assignment_challenge_2/` | Implement model training (StandardScaler, Keras model with L2 regularization, GroupKFold) and TFLite export, then deploy to ESP32. **Extra credit:** L1 regularization + Dropout in the model architecture, and enabling the extra credit features in the ESP32 inference code. |

## Progression

The assignments build on each other:

```
Tutorial          -> see the full pipeline working
Lab Challenge     -> implement export + ESP32 inference
Tech Challenge 1  -> implement feature engineering (required + extra credit)
Tech Challenge 2  -> implement training + export (uses your features from Challenge 1)
```

Each folder has its own `README.md` with detailed instructions and hints. Start there.

## Quick Start

Each folder is a self-contained project. To get started:

```bash
cd lab_tutorial
uv run main.py
```

Ensure you understand what is happening in `lab_tutorial` you will need it for completing the tasks after.

## Deliverables

**LOOK AT THE DELIVERABLES SECTIONS OF THE `README.md` FILES IN EACH FOLDER TO SEE WHAT YOU ARE EXPECTED TO SUBMIT!**
Please do not move the `README.md` files, and include the video for challenge 2 in the `tech_assignment_challenge_2/README.md`. 

## Note on ESP32 Compilation Warnings

When you build your C++ using PlatformIO you will see **many warning messages** scroll by — things like unused variables, deprecated APIs, and conversion warnings. **These are completely normal and expected.** The ESP32 toolchain and TFLite Micro library produce these by default. As long as the build ends with `[SUCCESS]`, you are fine. Only `error:` lines (not `warning:`) will cause a build failure.
