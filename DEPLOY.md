# Deploy ASL API so anyone can use the widget

Deploy this API once to **Render** (free tier). Then set that URL in the widget — **anyone** who opens the DeepSpace ASL widget will get the Python model with no installs or setup.

---

## 1. Put the API in its own repo (one-time)

Render needs a Git repo. Easiest:

1. Create a **new repo** on GitHub (e.g. `asl-sign-api`).
2. Copy **only** the contents of this `Sign-language-Recognition` folder into that repo:
   - `api.py`
   - `requirements.txt`
   - `Procfile`
   - `slr/` folder (entire folder, including `model/` with `slr_model.tflite` and `label.csv`)

So the repo root should have `api.py`, `requirements.txt`, `Procfile`, and `slr/` at the top level.

---

## 2. Create a Web Service on Render

1. Go to **https://render.com** and sign up (free).
2. **New** → **Web Service**.
3. Connect your GitHub account and select the repo you created.
4. Settings:
   - **Name**: e.g. `asl-sign-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python api.py`
   - **Plan**: Free
5. Click **Create Web Service**.

Render will build and deploy. The first build can take a few minutes (TensorFlow is large). When it’s done, you’ll get a URL like:

`https://asl-sign-api.onrender.com`

---

## 3. Wire the widget to the deployed URL

In the **ASL widget** `template.jsx`, set the default URL so everyone gets it automatically:

```js
const DEFAULT_SIGN_API_URL = 'https://YOUR-APP-NAME.onrender.com';
```

Replace with your real Render URL (no trailing slash, no `/classify-sign`).

Save and deploy/push the widget. After that, **anyone** who opens the ASL widget on that canvas will use the Python model with no extra steps.

---

## Notes

- **Free tier**: The service may sleep after ~15 min of no traffic. The first request after that can take 30–60 seconds; later requests are fast.
- **Model files**: Make sure `slr/model/slr_model.tflite` and `slr/model/label.csv` are in the repo so the API can load them.
