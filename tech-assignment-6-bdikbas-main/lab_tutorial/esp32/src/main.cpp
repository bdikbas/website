// Thermal Presence Detection — ESP32 Inference (Tutorial)
//
// Uses raw 64 pixel values + 1 BFS blob feature = 65 features.

#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_AMG88xx.h>


#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "model_params.h"

Adafruit_AMG88xx amg;
float pixels[AMG88xx_PIXEL_ARRAY_SIZE];


// TFLite globals
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Feature buffer: 64 raw pixels + 1 blob = 65
float features[N_FEATURES];

void setupModel() {
    model = tflite::GetModel(model_tflite);

    static tflite::AllOpsResolver resolver;
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &static_interpreter;

    interpreter->AllocateTensors();
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("[TFLite] Input: %d dims, type=%d\n",
                  input_tensor->dims->data[1], input_tensor->type);
    Serial.printf("[TFLite] Arena used: %d bytes\n",
                  interpreter->arena_used_bytes());
}

// BFS to find largest connected component of pixels > threshold in 8x8 grid
int largestBlob(float grid[8][8], float threshold) {
    bool visited[8][8] = {};
    int largest = 0;
    int qr[64], qc[64];

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (visited[r][c] || grid[r][c] <= threshold) continue;
            int size = 0;
            int head = 0, tail = 0;
            qr[tail] = r; qc[tail] = c; tail++;
            visited[r][c] = true;
            while (head < tail) {
                int cr = qr[head], cc = qc[head]; head++;
                size++;
                const int dr[] = {-1, 1, 0, 0};
                const int dc[] = {0, 0, -1, 1};
                for (int d = 0; d < 4; d++) {
                    int nr = cr + dr[d], nc = cc + dc[d];
                    if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8
                        && !visited[nr][nc] && grid[nr][nc] > threshold) {
                        visited[nr][nc] = true;
                        qr[tail] = nr; qc[tail] = nc; tail++;
                    }
                }
            }
            if (size > largest) largest = size;
        }
    }
    return largest;
}

void computeFeatures(float* raw_pixels, float* out_features) {
    // First 64 features: raw pixel values
    for (int i = 0; i < 64; i++) {
        out_features[i] = raw_pixels[i];
    }

    // Feature 65: largest blob via BFS
    float grid[8][8];
    for (int i = 0; i < 64; i++) grid[i / 8][i % 8] = raw_pixels[i];

    // Compute median (sort a copy) for threshold
    float sorted[64];
    memcpy(sorted, raw_pixels, 64 * sizeof(float));
    for (int i = 1; i < 64; i++) {
        float key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key) { sorted[j + 1] = sorted[j]; j--; }
        sorted[j + 1] = key;
    }
    float median = (sorted[31] + sorted[32]) / 2.0f;
    float threshold = median + 3.0f;

    out_features[64] = (float)largestBlob(grid, threshold);

    // Apply StandardScaler: (x - mean) / scale
    for (int i = 0; i < N_FEATURES; i++) {
        out_features[i] = (out_features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}

float runInference(float scaled_features[N_FEATURES]) {
    float input_scale = input_tensor->params.scale;
    int input_zero_point = input_tensor->params.zero_point;

    int8_t* input_data = input_tensor->data.int8;
    for (int i = 0; i < N_FEATURES; i++) {
        int val = (int)roundf(scaled_features[i] / input_scale) + input_zero_point;
        if (val < -128) val = -128;
        if (val > 127) val = 127;
        input_data[i] = (int8_t)val;
    }

    interpreter->Invoke();

    float output_scale = output_tensor->params.scale;
    int output_zero_point = output_tensor->params.zero_point;
    int8_t raw_output = output_tensor->data.int8[0];
    float confidence = (raw_output - output_zero_point) * output_scale;

    return confidence;
}

void setup() {
    Serial.begin(115200);
    delay(2000);

    Wire.begin();
    if (!amg.begin()) {
        Serial.println("[ERROR] AMG8833 not detected!");
        while (1) { delay(1000); }
    }

    setupModel();
    Serial.println("[OK] Model loaded, starting inference loop");
    delay(100);
}

void loop() {
    amg.readPixels(pixels);

    computeFeatures(pixels, features);
    float confidence = runInference(features);
    bool present = confidence > 0.5f;

    String message = "{\"prediction\":\"";
    message += present ? "present" : "empty";
    message += "\",\"confidence\":";
    message += String(confidence, 4);

    float maxTemp = pixels[0], minTemp = pixels[0];
    for (int i = 1; i < AMG88xx_PIXEL_ARRAY_SIZE; i++) {
        if (pixels[i] > maxTemp) maxTemp = pixels[i];
        if (pixels[i] < minTemp) minTemp = pixels[i];
    }
    message += ",\"max_temp\":";
    message += String(maxTemp, 1);
    message += ",\"min_temp\":";
    message += String(minTemp, 1);
    message += "}";

    // Serial.println(message); // Use for debugging if needed

    Serial.printf("[%s] conf=%.3f | min=%.1fC max=%.1fC\n",
                  present ? "PRESENT" : "EMPTY  ",
                  confidence, minTemp, maxTemp);

    delay(1000);
}
