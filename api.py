"""
Flask API for Sign Language Recognition model.
Exposes POST /classify-sign for the DeepSpace ASL widget.

Expects JSON: { "landmarks": [ [x0,y0], [x1,y1], ... ] }  (21 MediaPipe hand landmarks, normalized 0-1)
Returns JSON: { "letter": "A", "confidence": 0.95 } or { "letter": "", "confidence": 0 }

Run from Sign-language-Recognition directory:
  python api.py
Then use http://localhost:5000/classify-sign (or your deployed URL) in the ASL widget.
"""

import copy
import itertools
import os
import csv

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.after_request
def add_cors(resp):
    """Allow DeepSpace widget (and any origin) to call this API."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

# Paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "slr", "model", "slr_model.tflite")
LABEL_PATH = os.path.join(BASE_DIR, "slr", "model", "label.csv")

# Load TFLite model once at startup
interpreter = None
input_details = None
output_details = None
LABELS = []


def pre_process_landmark(landmark_list):
    """Same logic as slr.utils.pre_process: relative to wrist, flatten, normalize by max."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0.0, 0.0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = float(landmark_point[0]), float(landmark_point[1])
        temp_landmark_list[index][0] = float(temp_landmark_list[index][0]) - base_x
        temp_landmark_list[index][1] = float(temp_landmark_list[index][1]) - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value < 1e-9:
        max_value = 1.0
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list


def load_model_and_labels():
    global interpreter, input_details, output_details, LABELS
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open(LABEL_PATH, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        LABELS[:] = [row[0] for row in reader]
    print(f"[API] Loaded model and {len(LABELS)} labels: {LABELS}")


@app.route("/classify-sign", methods=["OPTIONS"])
def classify_sign_options():
    return "", 204


@app.route("/classify-sign", methods=["POST"])
def classify_sign():
    """Accept landmarks from MediaPipe (21 points, each [x,y] normalized 0-1). Return letter and confidence."""
    global interpreter, input_details, output_details, LABELS
    if interpreter is None:
        return jsonify({"letter": "", "confidence": 0, "error": "Model not loaded"}), 500

    data = request.get_json(silent=True) or {}
    landmarks = data.get("landmarks") or data.get("features")

    # Support both "landmarks" (21 pairs [x,y]) and legacy "features" (25-D) - for 25-D we cannot use this model
    if landmarks is None:
        return jsonify({"letter": "", "confidence": 0, "error": "Missing 'landmarks'"}), 400

    # Expect 21 points, each [x, y]
    if not isinstance(landmarks, (list, tuple)) or len(landmarks) != 21:
        return jsonify({"letter": "", "confidence": 0, "error": "Expected 21 landmarks [ [x,y], ... ]"}), 400

    try:
        landmark_list = []
        for pt in landmarks:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                landmark_list.append([float(pt[0]), float(pt[1])])
            elif isinstance(pt, dict) and "x" in pt and "y" in pt:
                landmark_list.append([float(pt["x"]), float(pt["y"])])
            else:
                return jsonify({"letter": "", "confidence": 0, "error": "Invalid landmark format"}), 400
        if len(landmark_list) != 21:
            return jsonify({"letter": "", "confidence": 0, "error": "Expected 21 landmarks"}), 400
    except (TypeError, ValueError) as e:
        return jsonify({"letter": "", "confidence": 0, "error": str(e)}), 400

    preprocessed = pre_process_landmark(landmark_list)
    preprocessed = np.array([preprocessed], dtype=np.float32)

    interpreter.set_tensor(input_details[0]["index"], preprocessed)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]["index"])
    result = np.squeeze(result)
    confidence = float(np.max(result))
    result_index = int(np.argmax(result))

    # Index 25 in the classifier means "below threshold" (no letter)
    if result_index >= len(LABELS) or result_index == 25 or confidence <= 0.5:
        return jsonify({"letter": "", "confidence": 0})

    letter = LABELS[result_index]
    return jsonify({"letter": letter, "confidence": round(confidence, 4)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": interpreter is not None})


if __name__ == "__main__":
    load_model_and_labels()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
