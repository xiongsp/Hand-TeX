import sqlite3
import json
import matplotlib.pyplot as plt
import sys


def visualize_symbol_instances(db_path, symbol_key=None):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # If no symbol key is provided, select random samples from different keys
    if symbol_key is None:
        cursor.execute("SELECT id, key, strokes FROM samples ORDER BY RANDOM() LIMIT 20")
        rows = cursor.fetchall()
        if not rows:
            print("No symbols found in the database.")
            conn.close()
            return
        print("No symbol key provided. Randomly selected different symbols.")
    else:
        # Query to fetch all instances of the symbol with the given key
        cursor.execute("SELECT id, strokes FROM samples WHERE key = ?", (symbol_key,))
        rows = cursor.fetchall()
        if not rows:
            print(f"No samples found for symbol key: {symbol_key}")
            conn.close()
            return
        print(f"Found {len(rows)} samples for symbol key: {symbol_key}")
    conn.close()

    # Loop over each instance and plot the strokes
    for idx, row in enumerate(rows):
        sample_id = row[0]
        strokes_data = row[-1]

        # Parse the strokes data from JSON
        try:
            strokes = json.loads(strokes_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for sample id {sample_id}: {e}")
            continue

        # Count total points in all strokes.
        total_points = sum(len(stroke) for stroke in strokes)
        if total_points < 2:
            continue

        # Enforce a plot range of 0-1000 for both axes.
        fig, axes = plt.subplots(2, 1, figsize=(5, 10))
        fig.suptitle(f"Sample ID: {sample_id} (Instance {idx + 1} of {len(rows)})")
        # Plot original strokes with points
        axes[0].set_title("With Points")
        for stroke in strokes:
            stroke_x = [point[0] for point in stroke]
            stroke_y = [point[1] for point in stroke]
            axes[0].plot(stroke_x, stroke_y, marker="o")

        # Plot original strokes without points
        for stroke in strokes:
            stroke_x = [point[0] for point in stroke]
            stroke_y = [point[1] for point in stroke]
            axes[1].plot(stroke_x, stroke_y)

        for i in range(2):
            axes[i].axis("equal")
            axes[i].set_xlabel("X Coordinate")
            axes[i].set_ylabel("Y Coordinate")
            axes[i].set_xlim(0, 1000)
            axes[i].set_ylim(1000, 0)
            axes[i].set_aspect("equal", adjustable="box")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()


if __name__ == "__main__":
    db_path = "../database/handtex.db"

    # Check if symbol key is provided as a command-line argument
    symbol_key = "stmaryrd-OT1-_ssearrow"

    if len(sys.argv) > 1:
        symbol_key = sys.argv[1]

    visualize_symbol_instances(db_path, symbol_key)
