"""
Flask API for Sign Language Recognition (DeepSpace ASL widget).

POST /classify-sign
  Body: {"landmarks": [[x0,y0],[x1,y1],...]}  (21 MediaPipe hand landmarks, normalized 0-1)
  Returns: {"letter": "A", "confidence": 0.95} or {"letter": "", "confidence": 0}

GET /health
GET /app  -> popup UI (camera + recognition)

Notes:
- This API does NOT use mediapipe; the widget sends landmarks.
- Uses the bundled TFLite model at slr/model/slr_model.tflite.
"""

import copy
import csv
import itertools
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)


@app.after_request
def add_cors(resp):
    """Allow browser calls from DeepSpace."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/", methods=["GET", "HEAD"])
def index():
    return jsonify(
        {
            "service": "sign-api-asl",
            "status": "ok",
            "endpoints": {
                "health": "/health",
                "classify": {"path": "/classify-sign", "method": "POST"},
                "app": "/app",
            },
        }
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "slr", "model", "slr_model.tflite")
LABEL_PATH = os.path.join(BASE_DIR, "slr", "model", "label.csv")

interpreter = None
input_details = None
output_details = None
LABELS = []


def pre_process_landmark(landmark_list):
    """Match slr.utils.pre_process.pre_process_landmark: wrist-relative, flatten 42, normalize."""
    temp = copy.deepcopy(landmark_list)

    base_x, base_y = 0.0, 0.0
    for idx, pt in enumerate(temp):
        if idx == 0:
            base_x, base_y = float(pt[0]), float(pt[1])
        pt[0] = float(pt[0]) - base_x
        pt[1] = float(pt[1]) - base_y

    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(list(map(abs, flat))) if flat else 1.0
    if max_val < 1e-9:
        max_val = 1.0

    return [v / max_val for v in flat]


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

    print(f"[API] Loaded model and {len(LABELS)} labels")


# Load model at import time so gunicorn has it ready
try:
    load_model_and_labels()
except Exception as e:
    print(f"[API] Model load failed: {e}")
    print(f"[API] Expected model at: {MODEL_PATH}")


@app.route("/classify-sign", methods=["OPTIONS"])
def classify_sign_options():
    return ("", 204)


@app.route("/classify-sign", methods=["POST"])
def classify_sign():
    if interpreter is None:
        return jsonify({"letter": "", "confidence": 0, "error": "Model not loaded"}), 500

    data = request.get_json(silent=True) or {}
    landmarks = data.get("landmarks")

    if not isinstance(landmarks, list) or len(landmarks) != 21:
        return jsonify({"letter": "", "confidence": 0, "error": "Expected 21 landmarks"}), 400

    try:
        pts = []
        for pt in landmarks:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pts.append([float(pt[0]), float(pt[1])])
            elif isinstance(pt, dict) and "x" in pt and "y" in pt:
                pts.append([float(pt["x"]), float(pt["y"])])
            else:
                return jsonify({"letter": "", "confidence": 0, "error": "Invalid landmark format"}), 400
    except Exception as e:
        return jsonify({"letter": "", "confidence": 0, "error": str(e)}), 400

    vec = pre_process_landmark(pts)
    x = np.array([vec], dtype=np.float32)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]["index"])
    out = np.squeeze(out)

    conf = float(np.max(out)) if out.size else 0.0
    idx = int(np.argmax(out)) if out.size else -1

    if conf <= 0.5 or idx < 0 or idx >= len(LABELS):
        return jsonify({"letter": "", "confidence": 0})

    return jsonify({"letter": LABELS[idx], "confidence": round(conf, 4)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": interpreter is not None})


@app.route("/app", methods=["GET"])
def app_page():
    """Serve the popup UI: camera + recognition, uses this API's /classify-sign."""
    static_dir = os.path.join(BASE_DIR, "static")
    app_html = os.path.join(static_dir, "app.html")
    if not os.path.isfile(app_html):
        return jsonify({"error": "app.html not found", "static_dir": static_dir}), 500
    return send_from_directory(static_dir, "app.html", mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
