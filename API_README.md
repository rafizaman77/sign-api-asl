# Sign Language API (for DeepSpace ASL widget)

This API wraps the Sign-language-Recognition TFLite model so the DeepSpace ASL widget can use it for classification.

## Run the API locally

From the `Sign-language-Recognition` directory:

```bash
pip install flask
python api.py
```

The server listens on **http://localhost:5000**.

- **POST /classify-sign** — expects JSON: `{ "landmarks": [ [x0,y0], [x1,y1], ... ] }` (21 MediaPipe hand landmarks, normalized 0–1). Returns `{ "letter": "A", "confidence": 0.95 }`.
- **GET /health** — returns `{ "status": "ok", "model_loaded": true }`.

## Connect to the DeepSpace ASL widget

1. **Option A – Same machine**: In the widget, set **Sign API URL** to `http://localhost:5000` only if you are loading the canvas from the same machine (e.g. `http://localhost:...`). Otherwise the browser will block cross-origin requests to localhost.

2. **Option B – Tunnel (e.g. ngrok)**  
   Expose your local API:
   ```bash
   ngrok http 5000
   ```
   In the widget, set **Sign API URL** to the HTTPS URL ngrok gives you (e.g. `https://abc123.ngrok.io`). Do **not** add `/classify-sign`; the widget adds it.

3. **Option C – Deploy**  
   Deploy this repo (e.g. Render, Railway) with:
   - **Build**: `pip install -r requirements.txt`
   - **Start**: `python api.py`  
   Set **Sign API URL** in the widget to your deployed base URL (e.g. `https://your-app.onrender.com`).

Once the URL is set, the widget sends hand landmarks to the API and uses the returned letter when its confidence is higher than the local classifier.
