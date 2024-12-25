import json
import time
import sqlite3
import sys
from itertools import chain
from math import sqrt

import matplotlib.pyplot as plt


# This can just classify basic shapes like circles, squares, triangles.
# It's used to find fit candidates for the inside similarity.


def is_good_circle(strokes: list[list[tuple[int, int]]]) -> tuple[bool, int]:
    """
    Returns True if the strokes form a good circle.
    If they do, the second return value is the side length of the largest square
    that fits inside the circle.
    The square naturally has its center at (500, 500).

    :param strokes: List of strokes, each stroke is a list of (x, y) points.
    :return: (is_circle, square_side)
    """
    strokes = resample_strokes(strokes, step=30.0)
    std_dev, min_radius = match_circle(strokes)
    optimal_square_side = fit_square_in_circle(min_radius)
    return min_radius >= 350 and std_dev <= 40, optimal_square_side


def is_good_square(strokes: list[list[tuple[int, int]]]) -> tuple[bool, int]:
    """
    Returns True if the strokes form a good square.
    If they do, the second return value is the side length of the largest square
    that fits inside the circle.
    The square naturally has its center at (500, 500).

    :param strokes: List of strokes, each stroke is a list of (x, y) points.
    :return: (is_square, square_side)
    """
    strokes = resample_strokes(strokes, step=30.0)
    side_error_stdev, min_linf = match_square(strokes)
    return min_linf >= 350 and side_error_stdev <= 20, min_linf


def is_good_triangle(strokes: list[list[tuple[int, int]]]) -> tuple[bool, int, int, int]:
    """
    Returns True if the strokes form a good triangle.
    The 3 other return values are the side length of the optimal square that fits inside the triangle,
    the center x coordinate of the square, and the center y coordinate of the square.
    The triangle's odd shape makes it necessary to shift the square's center.
    Note: unlike the other two methods, this one will likely produce an oversized square,
    so increase your security margin.

    :param strokes: List of strokes, each stroke is a list of (x, y) points.
    :return: (is_triangle, square_side, square_center_x, square_center_y)
    """
    strokes = resample_strokes(strokes, step=30.0)
    s, side_err_std = match_equilateral_triangle(strokes)
    square_side, square_center_x, square_center_y = fit_square_in_triangle(s, 500, 500)
    return s >= 900 and side_err_std <= 20, square_side, square_center_x, square_center_y


# ======================================== Internal stuffs =========================================


def fit_square_in_triangle(side_length: int, center_x: int, center_y: int) -> tuple[int, int, int]:
    """
    Returns (side_length, center_x, center_y) for the largest axis-aligned square
    that can fit inside an equilateral triangle with the given side length,
    centered at (cx, cy).
    """
    half_h = round((sqrt(3) / 4.0) * side_length)

    square_side = round((sqrt(3) * side_length) / (2.0 + sqrt(3)))

    square_cx = center_x
    square_cy = (center_y + half_h) - (square_side // 2)

    return square_side, square_cx, square_cy


def resample_strokes(
    strokes: list[list[tuple[int, int]]], step: float = 10.0
) -> list[list[tuple[int, int]]]:
    """
    Resample all strokes so that consecutive points are ~ 'step' units apart.

    :param strokes: A list of strokes, where each stroke is a list of (x, y) tuples.
    :param step: Desired spacing between consecutive points (in the same stroke).
    :return: A new list of strokes with uniformly spaced points.
    """
    resampled_strokes = []

    for stroke in strokes:
        # If the stroke has fewer than 2 points, we can't resample meaningfully
        if len(stroke) < 2:
            resampled_strokes.append(stroke)
            continue

        # The first point is always included
        resampled_stroke = [stroke[0]]

        # We'll keep track of how far we've traveled along the stroke
        accumulated_distance = 0.0

        for i in range(len(stroke) - 1):
            (x0, y0) = stroke[i]
            (x1, y1) = stroke[i + 1]

            # Calculate the segment length
            dx = x1 - x0
            dy = y1 - y0
            segment_length = sqrt(dx * dx + dy * dy)

            if segment_length == 0:
                continue  # two identical points in a row

            # We'll move along the segment, stepping by 'step' units
            distance_covered = 0.0

            while accumulated_distance + (segment_length - distance_covered) >= step:
                # We need to place a new point on this segment at 'step' from
                # the last resampled point

                # Figure out how far we must go from x0, y0 to add a new point
                distance_to_next_sample = step - accumulated_distance

                # Interpolate new point
                new_x = x0 + (dx * (distance_covered + distance_to_next_sample) / segment_length)
                new_y = y0 + (dy * (distance_covered + distance_to_next_sample) / segment_length)

                resampled_stroke.append((int(round(new_x)), int(round(new_y))))

                # Update distance_covered and accumulated_distance
                distance_covered += distance_to_next_sample
                accumulated_distance = 0.0  # We just placed a new point, so reset

            # We've reached the end of this segment; update how much more
            # distance is covered but didn't lead to a new point
            accumulated_distance += segment_length - distance_covered

        # Always include the last point of the stroke to ensure we end exactly
        # at the stroke's end
        if resampled_stroke[-1] != stroke[-1]:
            resampled_stroke.append(stroke[-1])

        resampled_strokes.append(resampled_stroke)

    return resampled_strokes


def match_circle(strokes: list[list[tuple[int, int]]]) -> tuple[float, float]:
    """
    Finds:
      1) std_dev: Standard deviation of the radii around the mean radius.
      2) min_radius: The minimum radius found in the data.

    :param strokes: List of strokes, each stroke is a list of (x, y) points.
    :return: (std_dev, min_radius)
    """

    center_x, center_y = 500, 500

    radii = []

    for x, y in chain.from_iterable(strokes):
        dx = x - center_x
        dy = y - center_y
        radius = sqrt(dx * dx + dy * dy)
        radii.append(radius)

    if len(radii) == 0:
        # Edge case: no points at all
        return 0.0, 0.0

    r_mean = sum(radii) / len(radii)

    # Standard deviation of radii around their mean
    sq_diff_sum = sum((r - r_mean) ** 2 for r in radii)
    std_dev = sqrt(sq_diff_sum / (len(radii) - 1))

    return std_dev, min(radii)


def fit_square_in_circle(r: int | float) -> int:
    """
    Returns the side length of the largest square that fits inside a circle of radius r.
    """
    # The square has a diagonal equal to the circle's diameter,
    # so the side length is sqrt(2) * r.
    return int(2 * r / sqrt(2))


def match_square(strokes: list[list[tuple[int, int]]]):
    """
    Return several measures of how well the points form a square centered at (500, 500):

      1. side_error_stdev: How far each point is from the side it is supposedly on.
         This is the standard deviation of those errors.
      2. min_linf: Minimum L∞ distance in the dataset (useful to see if points
         get too close to the center).

    :param strokes: List of strokes, each stroke is a list of (x, y) points
    :return: (side_error_stdev, min_linf)
    """

    center_x, center_y = 500, 500

    points = list(chain.from_iterable(strokes))

    if not points:
        return 0.0, 0.0, 0.0, 0.0

    # Compute L∞ distances for each point, this distance metric produces a unit square.
    linf_distances = []
    for x, y in points:
        linf_dist = max(abs(x - center_x), abs(y - center_y))
        linf_distances.append(linf_dist)

    mean_linf = sum(linf_distances) / len(linf_distances)
    min_linf = min(linf_distances)

    # Assign each point to the side it is closest to and compute error.
    side_errors = []
    left_x = center_x - mean_linf
    right_x = center_x + mean_linf
    top_y = center_y + mean_linf
    bottom_y = center_y - mean_linf

    for x, y in points:
        dx = x - center_x
        dy = y - center_y
        # Decide which side is "closest" by comparing |dx| to |dy|
        if abs(dx) > abs(dy):
            if dx < 0:
                error = abs(x - left_x)
            else:
                error = abs(x - right_x)
        else:
            if dy < 0:
                error = abs(y - bottom_y)
            else:
                error = abs(y - top_y)
        side_errors.append(error)

    # Standard deviation of side errors.
    side_error_mean = sum(side_errors) / len(side_errors)
    side_error_sq_diff_sum = sum((e - side_error_mean) ** 2 for e in side_errors)
    side_error_stdev = sqrt(side_error_sq_diff_sum / len(side_errors))

    return side_error_stdev, min_linf


def _dist_point_to_line_segment(px, py, x1, y1, x2, y2) -> float:
    """
    Returns the Euclidean distance from point p=(px, py) to
    the line segment connecting (x1, y1) and (x2, y2).
    """

    # Segment vector
    sx = x2 - x1
    sy = y2 - y1

    # Vector from (x1, y1) to p
    vx = px - x1
    vy = py - y1

    seg_len_squared = sx * sx + sy * sy
    if seg_len_squared == 0:
        # The segment's endpoints are the same point
        return sqrt(vx * vx + vy * vy)

    # Project vector v onto the segment [0..1]
    t = (vx * sx + vy * sy) / seg_len_squared

    # Clamp t to [0, 1] so that we stay within the segment
    if t < 0:
        t = 0
    elif t > 1:
        t = 1

    # Nearest point on the segment to p
    nx = x1 + t * sx
    ny = y1 + t * sy

    # Return distance from p to that nearest point
    dx = px - nx
    dy = py - ny
    return sqrt(dx * dx + dy * dy)


def match_equilateral_triangle(strokes: list[list[tuple[int, int]]]) -> tuple[int, float]:
    """
    Attempts to match an equilateral triangle whose bounding box
    is centered at (500, 500). We estimate the side length from the
    bounding box of the given strokes, then measure how close the points are
    to that ideal triangle.

    Returns a tuple with:
      1) side_length      : The chosen side length for the bounding-box-centered triangle.
      2) side_err_std     : The standard deviation of point->side distances.
    """

    center_x, center_y = 500, 500

    points = list(chain.from_iterable(strokes))
    if not points:
        return 0, 0.0

    # 1) Compute bounding box of the point set
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = x_max - x_min  # data width
    h = y_max - y_min  # data height

    # 2) Estimate side length s of an equilateral triangle
    #    bounding box. For an equilateral triangle:
    #      bounding_box_width  = s
    #      bounding_box_height = (sqrt(3)/2)*s
    #
    # => from width:  s_w = w
    # => from height: s_h = 2/√3 * h
    # We'll take a simple average, though you could do min, max, etc.
    s_w = w
    s_h = (2.0 / sqrt(3.0)) * h
    side_length = (s_w + s_h) / 2.0

    # 3) Build the triangle
    #    bounding box half-width = s/2
    #    bounding box half-height = (sqrt(3)/4)*s
    half_w = side_length / 2.0
    half_h = (sqrt(3) / 4.0) * side_length

    bottom_vertex = (center_x, center_y - half_h)
    left_vertex = (center_x - half_w, center_y + half_h)
    right_vertex = (center_x + half_w, center_y + half_h)

    segments = [
        (bottom_vertex, left_vertex),
        (left_vertex, right_vertex),
        (right_vertex, bottom_vertex),
    ]

    # 4) For each point, distance to nearest side
    side_distances = []
    for px, py in points:
        min_dist = float("inf")
        for (x1, y1), (x2, y2) in segments:
            d_seg = _dist_point_to_line_segment(px, py, x1, y1, x2, y2)
            if d_seg < min_dist:
                min_dist = d_seg
        side_distances.append(min_dist)

    mean_side_dist = sum(side_distances) / len(side_distances)
    var_side_dist = sum((d - mean_side_dist) ** 2 for d in side_distances) / len(side_distances)
    side_err_std = sqrt(var_side_dist)

    return side_length, side_err_std


def check_symbol_instances(db_path, symbol_key: str | list[str], shape: str = "circle"):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if isinstance(symbol_key, str):
        symbol_key = [symbol_key]
    # Query to fetch all instances of the symbol with the given key
    cursor.execute(
        "SELECT id, strokes FROM samples WHERE key IN ({})".format(",".join("?" * len(symbol_key))),
        symbol_key,
    )
    rows = cursor.fetchall()
    if not rows:
        print(f"No samples found for symbol key: {symbol_key}")
        conn.close()
        return
    print(f"Found {len(rows)} samples for symbol key: {symbol_key}")
    conn.close()

    # Print stats for the current shape, plotting all shape specific heuristics.
    results = []
    start_time = time.time()
    for row in rows:
        strokes_data = row[-1]
        try:
            strokes = json.loads(strokes_data)
            strokes = resample_strokes(strokes, step=30.0)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for sample id {row[0]}: {e}")
            continue
        if shape == "circle":
            std_dev, min_radius = match_circle(strokes)
            results.append((std_dev, min_radius))
        elif shape == "square":
            side_error_stdev, min_linf = match_square(strokes)
            results.append((side_error_stdev, min_linf))
        elif shape == "triangle":
            s, side_err_std = match_equilateral_triangle(strokes)
            results.append((s, side_err_std))
    print(f"Shape analysis took {1000 * (time.time() - start_time):.2f} ms")

    if shape == "circle":

        # Filter the results to remove those with less than 400 min radius, more than 40 std dev, more than 20 angle spread, and more than 50 largest gap.
        results = [result for result in results if result[1] >= 350 and result[0] <= 40]
        print(f"Filtered results: {len(results)}")
        # Plot a histogram for each stat.
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        fig.suptitle(f"Shape Analysis for {shape}")
        stats = ["Standard Deviation", "Minimum Radius"]
        for i, stat in enumerate(stats):
            ax = axes[i % 2]
            ax.set_title(stat)
            ax.hist([result[i] for result in results], bins=20)
            ax.set_xlabel(stat)
            ax.set_ylabel("Frequency")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    if shape == "square":
        # # Filter the results to remove those with less than 400 min radius, more than 40 std dev, more than 20 angle spread, and more than 50 largest gap.
        results = [result for result in results if result[1] >= 350 and result[0] <= 20]
        print(f"Filtered results: {len(results)}")
        # Plot a histogram for each stat.
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        fig.suptitle(f"Shape Analysis for {shape}")
        stats = ["Side Error Std Dev", "Min L∞"]
        for i, stat in enumerate(stats):
            ax = axes[i % 2]
            ax.set_title(stat)
            ax.hist([result[i] for result in results], bins=20)
            ax.set_xlabel(stat)
            ax.set_ylabel("Frequency")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    if shape == "triangle":
        # # Filter the results to remove those with less than 400 min radius, more than 40 std dev, more than 20 angle spread, and more than 50 largest gap.
        results = [result for result in results if result[1] <= 20 and result[0] >= 900]
        print(f"Filtered results: {len(results)}")
        # Plot a histogram for each stat.
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        fig.suptitle(f"Shape Analysis for {shape}")
        stats = ["Side Length", "Side Error Std Dev"]
        for i, stat in enumerate(stats):
            ax = axes[i % 2]
            ax.set_title(stat)
            ax.hist([result[i] for result in results], bins=20)
            ax.set_xlabel(stat)
            ax.set_ylabel("Frequency")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Loop over each instance and plot the strokes
    for idx, row in enumerate(rows):
        sample_id = row[0]
        strokes_data = row[-1]

        # Parse the strokes data from JSON
        try:
            strokes = json.loads(strokes_data)
            strokes = resample_strokes(strokes, step=30.0)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for sample id {sample_id}: {e}")
            continue

        classification_results = None
        title_infos = ""

        if shape == "circle":
            std_dev, min_radius = match_circle(strokes)
            title_infos = f"Std Dev: {std_dev:.2f}, Min Radius: {min_radius:.2f}"
            # Skip bad cirlces.
            if min_radius < 350 or std_dev > 40:
                continue
        elif shape == "square":
            side_error_stdev, min_linf = match_square(strokes)
            title_infos = f"Side Error Std Dev: {side_error_stdev:.2f}, Min L∞: {min_linf:.2f}"
            # Skip bad squares.
            if min_linf < 350 or side_error_stdev > 20:
                continue
        elif shape == "triangle":
            s, side_err_std = match_equilateral_triangle(strokes)
            title_infos = f"Side Length: {s:.2f}, Side Error Std Dev: {side_err_std:.2f}"
            # Skip bad triangles.
            if side_err_std > 20 or s < 900:
                continue

        # Enforce a plot range of 0-1000 for both axes.
        fig, axes = plt.subplots(2, 1, figsize=(5, 10))
        fig.suptitle(f"Sample ID: {sample_id} (Instance {idx + 1} of {len(rows)})\n{title_infos}")
        # Plot original strokes with points
        axes[0].set_title("With Points")
        for stroke in strokes:
            stroke_x = [point[0] for point in stroke]
            stroke_y = [point[1] for point in stroke]
            axes[0].plot(stroke_x, stroke_y, marker="o")

        # Resample the strokes to have points spaced 10 units apart
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

        if shape == "circle":
            # Draw the circle
            circle = plt.Circle((500, 500), min_radius, color="r", fill=False)
            axes[1].add_artist(circle)
            square_side = fit_square_in_circle(min_radius)
            square_center = (500, 500)
            top_left = (square_center[0] - square_side / 2, square_center[1] + square_side / 2)
            top_right = (square_center[0] + square_side / 2, square_center[1] + square_side / 2)
            bottom_left = (square_center[0] - square_side / 2, square_center[1] - square_side / 2)
            bottom_right = (square_center[0] + square_side / 2, square_center[1] - square_side / 2)
            axes[1].plot(
                [top_left[0], top_right[0], bottom_right[0], bottom_left[0], top_left[0]],
                [top_left[1], top_right[1], bottom_right[1], bottom_left[1], top_left[1]],
                color="g",
            )
        if shape == "square":
            # Draw the square
            left_x = 500 - min_linf
            right_x = 500 + min_linf
            top_y = 500 + min_linf
            bottom_y = 500 - min_linf
            axes[1].plot([left_x, right_x], [top_y, top_y], color="r")
            axes[1].plot([right_x, right_x], [top_y, bottom_y], color="r")
            axes[1].plot([right_x, left_x], [bottom_y, bottom_y], color="r")
            axes[1].plot([left_x, left_x], [bottom_y, top_y], color="r")
        if shape == "triangle":
            half_w = s / 2.0
            half_h = (sqrt(3) / 4.0) * s

            # For an upside-down triangle:
            #   bottom vertex is at y = cy - half_h
            #   top side spans [cx ± half_w, cy + half_h]
            #
            # So the three vertices are:
            bottom_vertex = (500, 500 - half_h)
            left_vertex = (500 - half_w, 500 + half_h)
            right_vertex = (500 + half_w, 500 + half_h)

            axes[1].plot(
                [bottom_vertex[0], left_vertex[0]], [bottom_vertex[1], left_vertex[1]], color="r"
            )
            axes[1].plot(
                [left_vertex[0], right_vertex[0]], [left_vertex[1], right_vertex[1]], color="r"
            )
            axes[1].plot(
                [right_vertex[0], bottom_vertex[0]], [right_vertex[1], bottom_vertex[1]], color="r"
            )

            # Plot the min square inside that.
            s_side, s_center_x, s_center_y = fit_square_in_triangle(s, 500, 500)
            s_side /= 2
            # s_center_y += s_side / 2
            top_left = (s_center_x - s_side, s_center_y + s_side)
            top_right = (s_center_x + s_side, s_center_y + s_side)
            bottom_left = (s_center_x - s_side, s_center_y - s_side)
            bottom_right = (s_center_x + s_side, s_center_y - s_side)
            axes[1].plot(
                [top_left[0], top_right[0], bottom_right[0], bottom_left[0], top_left[0]],
                [top_left[1], top_right[1], bottom_right[1], bottom_left[1], top_left[1]],
                color="g",
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()


if __name__ == "__main__":
    db_path = "database/handtex.db"

    # Check if symbol key is provided as a command-line argument
    # tipa-OT1-_textprimstress
    symbol_key = "latex2e-OT1-_textasciicircum"
    # 301
    if len(sys.argv) > 1:
        symbol_key = sys.argv[1]

    import handtex.symbol_relations as sr

    symbol_data = sr.SymbolData()

    symbol_key = symbol_data.get_similarity_group("latex2e-OT1-_bigcirc")
    shape = "circle"

    # symbol_key = symbol_data.get_similarity_group("amssymb-OT1-_square")
    # shape = "square"

    # symbol_key = symbol_data.get_similarity_group("latex2e-OT1-_bigtriangleup")
    # shape = "triangle"

    check_symbol_instances(db_path, symbol_key, shape)
