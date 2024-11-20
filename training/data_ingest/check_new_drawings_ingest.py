import sqlite3
import json
from pathlib import Path


def main():
    db_path = "../database/handtex.db"
    new_data_path = Path("../../new_drawings")

    # We want to check if for a given file, all the data is present in the database.
    # Calculate a percentage of overlap between the new data and the database.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for file in new_data_path.glob("*.json"):
        # Get the data from the file.
        with open(file, "r") as f:
            data = json.load(f)

        # Check if the data is present in the database.
        # The key is not unique, so we need to check if the strokes match.
        drawings_match = []
        for drawing in data:
            key = drawing["key"]
            strokes = drawing["strokes"]
            cursor.execute("SELECT * FROM samples WHERE key=?", (key,))
            db_data = cursor.fetchall()
            for db_drawing in db_data:
                db_strokes = json.loads(db_drawing[2])
                if strokes == db_strokes:
                    drawings_match.append(drawing)

        # Calculate the percentage of overlap.
        percentage_overlap = len(drawings_match) / len(data) * 100
        if percentage_overlap == 0:
            print(f"No overlap for {file.name}")
        elif percentage_overlap == 100:
            print(f"Full overlap for {file.name}")
        else:
            print(f"Partial overlap for {file.name}: {percentage_overlap:.2f}%")
            for drawing in drawings_match:
                print(drawing)
            print("\n")

    conn.close()


if __name__ == "__main__":
    main()
