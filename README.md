# IoT Fall Detection Server

Flask API that reads one IMU sample from Firebase Realtime Database, runs the CNN+LSTM model, and writes `fallDetected=true` back to Firebase when a fall class is predicted.

## What This Repo Contains

- `app.py` - Flask API + poller loop
- `inference.py` - CNN+LSTM inference (`.keras` + scaler + label encoder)
- `firebase_rtdb.py` - Firebase REST client
- `config.py` - env-based configuration
- `api/index.py` + `vercel.json` - Vercel deployment entrypoint/routing
- `scripts/run_model_once.py` - local one-sample inference check

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy env template and fill values:

```bash
copy .env.example .env
```

Required files in `models/`:

- `fall_detection_model.keras`
- `label_encoder.pkl`
- `scaler.pkl`

## 2) Environment Variables

Read from `.env` (local) or platform env vars:

- `FIREBASE_HOST`
- `FIREBASE_SECRET`
- `FIREBASE_VITALS_PATH` (default `patient/vitals`)
- `POLL_INTERVAL_SEC` (default `4`)
- `ENABLE_POLLER` (`true/false`)
- `FALL_KERAS_MODEL_PATH`
- `LABEL_ENCODER_PATH`
- `SCALER_PATH`
- `FALL_MODEL_TIMESTEPS` (default `200`)
- `FALL_ACTIVITY_LABELS` (default `fall,lfall,rfall`)

## 3) Run Locally

Start API + poller:

```bash
python app.py
```

Manual sync:

```bash
curl http://127.0.0.1:5000/api/sync
```

One-sample local inference:

```bash
python scripts/run_model_once.py
```

## 4) API

- `GET /health` - health check
- `GET /api/status` - last prediction/error info
- `GET /api/vitals` - raw Firebase vitals
- `GET /api/sync` - fetch vitals, infer, patch fall flag if needed
- `POST /api/sync` with `{ "force": true }` - same sync trigger

Example:

```bash
curl http://127.0.0.1:5000/api/status
```

## 5) Vercel Deploy

This repo is prepared with:

- `api/index.py` exporting `app`
- `vercel.json` using `@vercel/python` and routing all paths to `api/index.py`

Deploy:

```bash
vercel
```

Set env vars in Vercel Project Settings:

- `FIREBASE_HOST`
- `FIREBASE_SECRET`
- `FIREBASE_VITALS_PATH`
- `FALL_KERAS_MODEL_PATH`
- `LABEL_ENCODER_PATH`
- `SCALER_PATH`
- `FALL_MODEL_TIMESTEPS`
- `FALL_ACTIVITY_LABELS`

Then call your deployed API:

```bash
curl https://<your-vercel-domain>/api/sync
```

## Notes

- Polling every 4s only happens when the Python process is running continuously (local/container).  
  On serverless platforms, prefer external schedulers hitting `/api/sync`.
- Keep `.env` private. Only commit `.env.example`.
