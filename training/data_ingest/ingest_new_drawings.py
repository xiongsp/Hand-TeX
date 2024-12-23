import sqlite3
import json
from pathlib import Path
from rdp import rdp


def main():
    db_path = "../database/handtex.db"
    new_data_path = Path("../../new_drawings")

    """
    CREATE TABLE samples (
        id INTEGER PRIMARY KEY,
        key TEXT,
        strokes TEXT
    )
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a checkpoint to roll back to in case of an error.
    cursor.execute("BEGIN TRANSACTION")

    drawings = 0

    for file in new_data_path.glob("*.json"):
        # Get the data from the file.
        with open(file, "r") as f:
            data = json.load(f)

        # Insert the data into the database.
        for drawing in data:
            key = drawing["key"]
            strokes = drawing["strokes"]
            # strokes = [rdp(stroke, epsilon=6) for stroke in strokes]
            cursor.execute(
                "INSERT INTO samples (key, strokes) VALUES (?, ?)", (key, json.dumps(strokes))
            )
            drawings += 1

    # Commit the transaction.
    conn.commit()
    conn.close()

    print(f"Added {drawings} drawings to the database.")


if __name__ == "__main__":
    main()
