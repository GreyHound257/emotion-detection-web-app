# Emotion Detection Web App

## Overview
This is a web application for detecting emotions (angry, disgust, fear, happy, neutral, sad, surprise) from facial images using a PyTorch-based CNN model. It supports webcam capture or image uploads, stores predictions in a SQLite database, and displays results.

## Dataset
- Used: FER-2013 (download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013).
- Structure: Images in `data/` subfolders by emotion.
- Training: ~28k images (train split).
- Testing: ~7k images (test split).
- Preprocessing: Grayscale, resized to 48x48, normalized.

## Model
- Architecture: Custom CNN with convolutional layers, max pooling, dropout, and fully connected layers.
- Training: Data augmentation (rotation, flip), Adam optimizer, cross-entropy loss. Hyperparameter tuning via basic grid search (learning rates: [0.001, 0.0001]; batch sizes: [32, 64]).
- Evaluation: On separate test set – accuracy, precision, recall, F1, confusion matrix.
- Saved model: `trained_models/groovy_emotion_detector_v1.pth` (creative name; .pth for PyTorch).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Initialize DB: `python init_database.py`
3. Train model (optional, if retraining): `python model.py`
4. Run app: `python app.py`
5. Access: http://localhost:5000

## Training from Scratch
- Run `model.py` locally or on Google Colab (see Colab instructions below).
- Enhancements: Data augmentation, early stopping, hyperparameter tuning.
- Evaluation: Prints metrics and saves confusion matrix plot as `confusion_matrix.png`.

## Database
- SQLite: `database.sqlite3`
- Tables: `users` (id, username), `predictions` (id, user_id, image_path, predicted_emotion, timestamp).
- Integration: Stores user (dummy for now) and prediction data.

## Web Interface
- `index.html`: Webcam capture or upload, submit for prediction.
- `result.html`: Displays image, predicted emotion, and link to view past predictions.
- Styling: Basic CSS for better UX.
- Error Handling: Catches invalid images, server errors.

## Deployment
- Push to GitHub.
- Deploy to Render.com (connect repo, set Python runtime).
- Web app link: [See link_to_my_web_app.txt]

## Colab for Retraining
Copy `model.py` to a Colab notebook, upload dataset, run the script. Install deps via !pip.

## Limitations/Improvements
- Accuracy: ~65% on test set (typical for FER-2013; improve with more data/pretrained models like ResNet).
- Expand testing: Added per-class metrics.
- Security: Add auth for production.