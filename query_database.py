import sqlite3

def get_recent_predictions():
    conn = sqlite3.connect('database.sqlite3')
    cursor = conn.cursor()
    cursor.execute("SELECT emotion, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 10")
    predictions = cursor.fetchall()
    conn.close()
    import pandas as pd
    return pd.DataFrame(predictions, columns=['emotion', 'timestamp'])

# Example usage (for testing)
if __name__ == "__main__":
    print(get_predictions())