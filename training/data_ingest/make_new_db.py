import sqlite3

conn = sqlite3.connect("handtex.db")
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE samples (
        id INTEGER PRIMARY KEY,
        key TEXT,
        strokes TEXT
    )
"""
)

# Add an index to the key column for faster lookups.
cursor.execute("CREATE INDEX key_index ON samples (key)")


conn.close()
