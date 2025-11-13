import sqlite3

def get_recent_predictions():
    conn = sqlite3.connect('database.sqlite3')
    cursor = conn.cursor()
    cursor.execute("SELECT emotion, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 10")
    predictions = cursor.fetchall()
    conn.close()
    import pandas as pd
    return pd.DataFrame(predictions, columns=['emotion', 'timestamp'])

def save_prediction(emotion):
    conn = sqlite3.connect('database.sqlite3')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (emotion) VALUES (?)", (emotion,))
    conn.commit()
    conn.close()