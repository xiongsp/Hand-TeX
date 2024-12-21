import sqlite3
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from rdp import rdp
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

db_in_path = "../database/handtex.db"
db_out_path = "../database/handtex_processed.db"

# Nuke the output database if it exists
import os

if os.path.exists(db_out_path):
    os.remove(db_out_path)


def plot_stroke_pair(s1, s2, symbol_name):
    # We want to plot 4 plots: s1 with points, s1 without points, s2 with points, s2 without points

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Enforce a plot range of 0-1000 for both axes.
    fig.suptitle(f"Sample name: {symbol_name}")
    # Plot the cleaned strokes with points
    axes[0, 0].set_title("With Points")
    for stroke in s1:
        stroke_x = [point[0] for point in stroke]
        stroke_y = [point[1] for point in stroke]
        axes[0, 0].plot(stroke_x, stroke_y, marker="o")

    # Plot the cleaned strokes without points
    for stroke in s1:
        stroke_x = [point[0] for point in stroke]
        stroke_y = [point[1] for point in stroke]
        axes[1, 0].plot(stroke_x, stroke_y)

    # Plot the original strokes with points
    axes[0, 1].set_title("With Points")
    for stroke in s2:
        stroke_x = [point[0] for point in stroke]
        stroke_y = [point[1] for point in stroke]
        axes[0, 1].plot(stroke_x, stroke_y, marker="o")

    # Plot the original strokes without points
    for stroke in s2:
        stroke_x = [point[0] for point in stroke]
        stroke_y = [point[1] for point in stroke]
        axes[1, 1].plot(stroke_x, stroke_y)

    for i in range(2):
        for j in range(2):
            axes[i, j].axis("equal")
            axes[i, j].set_xlabel("X Coordinate")
            axes[i, j].set_ylabel("Y Coordinate")
            axes[i, j].set_xlim(0, 1000)
            axes[i, j].set_ylim(1000, 0)
            axes[i, j].set_aspect("equal", adjustable="box")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


input_conn = sqlite3.connect(db_in_path)
input_cursor = input_conn.cursor()

# Create a new database and table for the cleaned data
output_conn = sqlite3.connect(db_out_path)
output_cursor = output_conn.cursor()

output_cursor.execute(
    """
    CREATE TABLE samples (
        id INTEGER PRIMARY KEY,
        key TEXT,
        strokes TEXT
    )
"""
)
output_cursor.execute("CREATE INDEX key_index ON samples (key)")

# Fetch all rows from the resampled database
input_cursor.execute("SELECT * FROM samples")
rows = input_cursor.fetchall()


total_points = 0
total_points_cleaned = 0

counts: dict[str, list[int]] = defaultdict(list)

# Perform a first pass to figure out the std. deviation over the number of points per symbol class.
for row in rows:
    _, key, strokes_json = row
    strokes: list[list[tuple[int, int]]] = json.loads(strokes_json)
    points = sum(len(stroke) for stroke in strokes)
    counts[key].append(points)

thresholds_min_points = {}

for key, point_counts in counts.items():
    if len(point_counts) < 2:
        thresholds_min_points[key] = point_counts[0]
        continue

    mean_points = sum(point_counts) / len(point_counts)
    std_dev = (sum((x - mean_points) ** 2 for x in point_counts) / (len(point_counts) - 1)) ** 0.5

    thresholds_min_points[key] = int(mean_points + std_dev)

# Print a sorted list of the thresholds.
for key, threshold in sorted(thresholds_min_points.items(), key=lambda x: x[1]):
    print(f"{key}: {threshold}")


def operation(s_id, symbol_key, s):
    point_count = sum(len(stroke) for stroke in s)
    global total_points
    global total_points_cleaned
    total_points += point_count
    if point_count < thresholds_min_points[symbol_key] and s_id <= 339935:
        total_points_cleaned += point_count
        return s

    simple_strokes = [rdp(stroke, epsilon=6) for stroke in s]

    # If we don't have an improvement, smooth harder. Add a constant offset for small strokes.
    simple_count = sum(len(stroke) for stroke in simple_strokes)
    tag = f"({point_count} p - {1- simple_count/point_count:.2%})"
    if simple_count >= 0.6 * point_count + 5:
        simple_strokes = [rdp(stroke, epsilon=10) for stroke in s]
        simple_count = sum(len(stroke) for stroke in simple_strokes)
        tag = f" ({point_count} p - {1 -simple_count/point_count:.2%} SUPER)"

    # Plot it to see what it looks like.
    # plot_stroke_pair(simple_strokes, s, f"{symbol_key} (ID: {s_id}) {tag}")

    total_points_cleaned += simple_count

    return simple_strokes


# Process each row, removing consecutive duplicate coordinates
for row in tqdm(rows):
    sample_id, key, strokes_json = row
    strokes = json.loads(strokes_json)
    op_strokes = operation(sample_id, key, strokes)
    op_strokes_json = json.dumps(op_strokes)
    # Insert the cleaned data into the new database
    output_cursor.execute(
        "INSERT INTO samples (id, key, strokes) VALUES (?, ?, ?)",
        (sample_id, key, op_strokes_json),
    )

print(f"Total points before cleaning: {total_points}")
print(f"Total points after cleaning: {total_points_cleaned}")
print(
    f"Total points removed: {total_points - total_points_cleaned} ({(total_points - total_points_cleaned) / total_points:.2%})"
)

# Commit changes and close connections
output_conn.commit()
input_conn.close()
output_conn.close()
