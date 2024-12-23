import sqlite3
import json
from pathlib import Path


def matching_strokes(strokes_db: list[list[list[int]]], strokes_new: list[list[list[int]]]) -> bool:
    """
    This check should account for possible path simplifications that took place,
    which only includes dropping points, not adding or altering them.
    So we just need to check if the number of strokes is the same,
    if the first and last point are a precise match, and the set of points is a
    subset of the db set.

    :param strokes_db:
    :param strokes_new:
    :return: True if close enough to be considered a match.
    """

    if len(strokes_db) != len(strokes_new):
        return False

    for stroke_db, stroke_new in zip(strokes_db, strokes_new):
        if stroke_db[0] != stroke_new[0] or stroke_db[-1] != stroke_new[-1]:
            return False

        # # To be able to hash the stroke, we need to turn the inner coordinates into tuples.
        # stroke_db = [tuple(point) for point in stroke_db]
        # stroke_new = [tuple(point) for point in stroke_new]
        # if not set(stroke_new).issubset(set(stroke_db)):
        #     return False

    return True


def main():
    db_path = "../database/handtex.db"
    new_data_path = Path("../../new_drawings")

    # We want to check if for a given file, all the data is present in the database.
    # Calculate a percentage of overlap between the new data and the database.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for file in sorted(new_data_path.glob("*.json")):
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
                if matching_strokes(db_strokes, strokes):
                    drawings_match.append(drawing)
                    break

        # Calculate the percentage of overlap.
        percentage_overlap = len(drawings_match) / len(data) * 100
        if percentage_overlap == 0:
            print(f"No overlap for {file.name}")
        elif percentage_overlap >= 100:
            print(f"Full overlap for {file.name}")
        else:
            print(f"Partial overlap for {file.name}: {percentage_overlap:.2f}%")
            for drawing in drawings_match:
                print(drawing)
            print("\n")

    conn.close()


if __name__ == "__main__":
    main()
