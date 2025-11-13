import sqlite3
import os

# Connect to database
conn = sqlite3.connect('database.sqlite3')
cursor = conn.cursor()

# Drop existing table if it exists (to recreate with correct schema)
cursor.execute("DROP TABLE IF EXISTS predictions")

# Create new table with correct columns
cursor.execute('''CREATE TABLE predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  emotion TEXT,
                  image_path TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

conn.commit()
conn.close()
print("Database reinitialized with correct schema.")