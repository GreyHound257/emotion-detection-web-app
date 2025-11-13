import sqlite3
import pandas as pd

def save_prediction(name, emotion, image_path):
    try:
        conn = sqlite3.connect('database.sqlite3')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (name, emotion, image_path) VALUES (?, ?, ?)", (name, emotion, image_path))
        conn.commit()
        conn.close()
        print(f"Saved prediction for {name}: {emotion}")
    except sqlite3.OperationalError as e:
        print(f"Database error in save_prediction: {e}")
        raise

def get_recent_predictions():
    try:
        conn = sqlite3.connect('database.sqlite3')
        cursor = conn.cursor()
        cursor.execute("SELECT name, emotion, image_path, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 10")
        predictions = cursor.fetchall()
        conn.close()
        df = pd.DataFrame(predictions, columns=['name', 'emotion', 'image_path', 'timestamp'])
        return df
    except sqlite3.OperationalError as e:
        print(f"Database error in get_recent_predictions: {e}")
        return pd.DataFrame()  # Return empty DataFrame to avoid crash