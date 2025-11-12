import sqlite3

def get_predictions(user_id=1):
    conn = sqlite3.connect('database.sqlite3')
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, predicted_emotion, timestamp FROM predictions WHERE user_id = ?", (user_id,))
    results = cursor.fetchall()
    conn.close()
    return results

# Example usage (for testing)
if __name__ == "__main__":
    print(get_predictions())