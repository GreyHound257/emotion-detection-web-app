# Emotion Detection Web App

This project is a Flask web application that can detect a person's emotion from an uploaded image (or a live capture sent to the server). It includes model training, evaluation on a separate test set, and SQLite database integration for storing predictions.

Key files:
- `app.py` — Flask application, handles uploads and prediction requests.
- `model.py` — Model building, training, saving, loading, evaluation and prediction routines. Evaluation is done on a separate `test` dataset.
- `init_database.py` — Create the SQLite database and required table.
- `query_database.py` — Simple query helpers to fetch stored predictions.
- `requirements.txt` — Python dependencies.
- `templates/` — `index.html` and `result.html` for the UI.
- `static/css/style.css` — Simple styling.
- `trained_models/` — Saved model files (the trained model will be saved here, e.g., `ember_emotion_v1.h5`).

Data layout expected (one of these):
- Option A (preferred):
  - `data/train/<class>/*`, `data/val/<class>/*`, `data/test/<class>/*`
- Option B (single folder):
  - `data/<class>/*` and the code will create a train/val/test split automatically if `data/test` is missing.

How to use
1. Create a virtual environment and install requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\activate; pip install -r requirements.txt
```

2. Prepare your image dataset under `data/` as described above. If you only have `data/<class>/*`, run `model.py` to split and create test/val sets.

3. (Optional) Train the model:

```powershell
python model.py --train --epochs 10 --batch_size 32
```

4. Run the Flask app:

```powershell
python app.py
```

Notes
- The model will be saved into `trained_models/` as `ember_emotion_v1.h5` by default.
- Evaluation metrics (accuracy and classification report) are produced on a separate test set to avoid data leakage.

Next steps / improvements
- Add Keras Tuner-based hyperparameter tuning (optional dependency commented in `requirements.txt`).
- Add webcam frontend (getUserMedia) with base64 upload to the Flask endpoint.
- Add unit tests for model preprocessing and DB helpers.
