import sqlite3
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from rdp import rdp
from tqdm import tqdm
from training.shape_classifier import resample_strokes
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


total_strokes = 0
total_strokes_cleaned = 0

counts: dict[str, list[int]] = defaultdict(list)

# Perform a first pass to figure out the std. deviation over the number of points per symbol class.
for row in rows:
    _, key, strokes_json = row
    strokes: list[list[tuple[int, int]]] = json.loads(strokes_json)
    counts[key].append(len(strokes))

thresholds_min_points = {}

for key, point_counts in counts.items():
    if len(point_counts) < 2:
        thresholds_min_points[key] = point_counts[0]
        continue

    mean_points = sum(point_counts) / len(point_counts)
    std_dev = (sum((x - mean_points) ** 2 for x in point_counts) / (len(point_counts) - 1)) ** 0.5

    thresholds_min_points[key] = int(mean_points + 5 * std_dev)

# Print a sorted list of the thresholds.
for key, threshold in sorted(thresholds_min_points.items(), key=lambda x: x[1]):
    print(f"{key}: {threshold}")


# def operation(s_id, symbol_key, s):
#     point_count = sum(len(stroke) for stroke in s)
#     global total_points
#     global total_points_cleaned
#     total_points += point_count
#     if (
#         key != "latex2e-/"
#         or point_count < thresholds_min_points[symbol_key]
#         or point_count < 15
#         or s_id > 339935
#     ):
#         total_points_cleaned += point_count
#         return s
#
#     simple_strokes = [rdp(stroke, epsilon=6) for stroke in s]
#
#     # If we don't have an improvement, smooth harder. Add a constant offset for small strokes.
#     simple_count = sum(len(stroke) for stroke in simple_strokes)
#     tag = f"({point_count} p - {1- simple_count/point_count:.2%})"
#     if simple_count >= 0.6 * point_count + 5:
#         simple_strokes = [rdp(stroke, epsilon=10) for stroke in s]
#         simple_count = sum(len(stroke) for stroke in simple_strokes)
#         tag = f" ({point_count} p - {1 -simple_count/point_count:.2%} SUPER)"
#
#     # Plot it to see what it looks like.
#     # plot_stroke_pair(simple_strokes, s, f"{symbol_key} (ID: {s_id}) {tag}")
#
#     total_points_cleaned += simple_count
#
#     return simple_strokes

import handtex.symbol_relations as sr

symbol_data = sr.SymbolData()


def operation(s_id, symbol_key, s):
    if symbol_key not in symbol_data.get_similarity_group(
        #     "latex2e-_guilsinglright"
        # ) + symbol_data.get_similarity_group(
        #     "latex2e-_guilsinglleft"
        #     ) + symbol_data.get_similarity_group(
        "latex2e-_|"
    ) + symbol_data.get_similarity_group("amssymb-_intercal"):
        return s

    # # We want to squish the strokes to fit within a width of 400.
    # min_y = min(min(point[1] for point in stroke) for stroke in s)
    # max_y = max(max(point[1] for point in stroke) for stroke in s)
    # We want to squish the strokes to fit within a width of 400.
    min_x = min(min(point[0] for point in stroke) for stroke in s)
    max_x = max(max(point[0] for point in stroke) for stroke in s)

    # height = max_y - min_y
    width = max_x - min_x
    scaled = False
    # if height > 400:
    if width > 600:
        s_old = s
        scaled = True
        scale = 0.65
        # We need to keep it centered on a 1000x1000 canvas.
        # offset = (1000 - height * scale) / 2
        # s_new = [
        #     [(int(point[0]), int((point[1] - min_y) * scale + offset)) for point in stroke]
        #     for stroke in s
        # ]
        offset = (1000 - width * scale) / 2
        s_new = [
            [(int((point[0] - min_x) * scale + offset), int(point[1])) for point in stroke]
            for stroke in s
        ]
        s = [rdp(stroke, epsilon=3) for stroke in s_new]
        # Plot it to see what it looks like.
        plot_stroke_pair(s, s_old, f"{symbol_key} (ID: {s_id}) {'(scaled)' if scaled else ''}")

    return s


def stroke_has_stair_steps(stroke: list[list[list[int]]]) -> bool:
    """
    We want to detect when the stroke has a series of stair steps in it,
    which resulted from scaling the stroke.

    Stair steps are defined as a series of 3 points that form a perfect right angle.

    :param stroke: a list of strokes, which are a list of (x, y) coordinates.
    :return: True is the stroke has stair steps, False otherwise.
    """
    min_step_count = 3
    min_step_size = 8
    max_step_size = 40

    for stroke in stroke:
        steps = 0
        for i in range(2, len(stroke)):
            x0, y0 = stroke[i - 2]
            x1, y1 = stroke[i - 1]
            x2, y2 = stroke[i]

            # Check if the points form a right angle.
            if (x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) == 0:
                # Check if the step size is within the threshold.
                if (
                    abs(x1 - x0) < max_step_size
                    and abs(y1 - y0) < max_step_size
                    and abs(x2 - x1) < max_step_size
                    and abs(y2 - y1) < max_step_size
                ):
                    if (
                        abs(x1 - x0) > min_step_size
                        or abs(y1 - y0) > min_step_size
                        or abs(x2 - x1) > min_step_size
                        or abs(y2 - y1) > min_step_size
                    ):
                        steps += 1

            if steps >= min_step_count:
                return True
    return False


# TODO fix the parallel symbol being slanty
def operation_fix_stairs(s_id, symbol_key, s):
    if not stroke_has_stair_steps(s):
        return s

    # Perform resampling to create more intermediate points.
    resampled_s = resample_strokes(s, 3)
    # Perform smoothing to remove the stair steps.
    smoothed_s = [rdp(stroke, epsilon=7) for stroke in resampled_s]

    # Get difference if point counts.
    total_points = sum(len(stroke) for stroke in s)
    total_points_resampled = sum(len(stroke) for stroke in resampled_s)
    total_points_smoothed = sum(len(stroke) for stroke in smoothed_s)

    # assert total_points_smoothed <= total_points

    # Just show it for now.
    if total_points_smoothed > total_points:
        # plot_stroke_pair(
        #     s,
        #     smoothed_s,
        #     f"{symbol_key} (ID: {s_id}) ({total_points} -> {total_points_resampled} -> {total_points_smoothed} [{1- total_points_smoothed/total_points:.2%}])",
        # )
        print(
            f"{symbol_key} (ID: {s_id}) ({total_points} -> {total_points_resampled} -> {total_points_smoothed} [{1- total_points_smoothed/total_points:.2%}])"
        )

        return s
    return smoothed_s


# Process each row, removing consecutive duplicate coordinates
for row in tqdm(rows):
    sample_id, key, strokes_json = row
    strokes = json.loads(strokes_json)
    op_strokes = operation_fix_stairs(sample_id, key, strokes)
    op_strokes_json = json.dumps(op_strokes)
    # Insert the cleaned data into the new database
    output_cursor.execute(
        "INSERT INTO samples (id, key, strokes) VALUES (?, ?, ?)",
        (sample_id, key, op_strokes_json),
    )

print(f"Total points before cleaning: {total_strokes}")

# Commit changes and close connections
output_conn.commit()
input_conn.close()
output_conn.close()
