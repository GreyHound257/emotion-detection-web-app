import sqlite3
import pandas as pd

def save_prediction(name, emotion, image_path):
    conn = sqlite3.connect('database.sqlite3')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (name, emotion, image_path) VALUES (?, ?, ?)", (name, emotion, image_path))
    conn.commit()
    conn.close()

def get_recent_predictions():
    conn = sqlite3.connect('database.sqlite3')
    cursor = conn.cursor()
    cursor.execute("SELECT name, emotion, image_path, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 10")
    predictions = cursor.fetchall()
    conn.close()
    return pd.DataFrame(predictions, columns=['name', 'emotion', 'image_path', 'timestamp'])