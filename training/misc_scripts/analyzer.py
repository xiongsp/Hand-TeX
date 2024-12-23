import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial import distance


def analyze_symbol_data(db_path, symbol_key=None | str | list[str]):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # If no symbol key is provided, analyze all available keys
    if symbol_key is None:
        cursor.execute("SELECT DISTINCT key FROM samples")
        keys = [row[0] for row in cursor.fetchall()]
        if not keys:
            print("No symbols found in the database.")
            conn.close()
            return
    elif isinstance(symbol_key, str):
        keys = [symbol_key]
    else:
        keys = symbol_key

    all_x_coords = []
    all_y_coords = []
    num_points_per_stroke = []
    num_coordinates_per_sample = []
    bounding_box_sizes = []
    num_float_coords = 0
    total_coords = 0
    distances = []

    min_points_sample = None
    max_points_sample = None
    min_bbox_sample = None
    max_bbox_sample = None
    narrowest_sample = None
    widest_sample = None
    tallest_sample = None
    shortest_sample = None
    smallest_max_dim_sample = None

    min_points = float("inf")
    max_points = 0
    min_bbox_area = float("inf")
    max_bbox_area = 0
    min_coordinates = float("inf")
    max_coordinates = 0
    min_width = float("inf")
    max_width = 0
    min_height = float("inf")
    max_height = 0
    min_max_dim = float("inf")

    # Variables to remember extreme figures for the entire dataset
    dataset_min_points_sample = None
    dataset_max_points_sample = None
    dataset_min_bbox_sample = None
    dataset_max_bbox_sample = None
    dataset_narrowest_sample = None
    dataset_widest_sample = None
    dataset_tallest_sample = None
    dataset_shortest_sample = None
    dataset_smallest_max_dim_sample = None

    for key in keys:
        print(f"Analyzing data for symbol key: {key}")

        # Query to fetch all instances of the symbol with the given key
        cursor.execute("SELECT id, strokes FROM samples WHERE key = ?", (key,))
        rows = cursor.fetchall()

        if not rows:
            print(f"No samples found for symbol key: {key}")
            continue

        print(f"Found {len(rows)} samples for symbol key: {key}")

        # Loop over each instance to collect data
        for idx, row in enumerate(rows):
            sample_id = row[0]
            strokes_data = row[1]

            # Parse the strokes data from JSON
            try:
                strokes = json.loads(strokes_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for sample id {sample_id}: {e}")
                continue

            sample_x_coords = []
            sample_y_coords = []
            total_points = 0
            for stroke in strokes:
                stroke_x = [point[0] for point in stroke]
                stroke_y = [point[1] for point in stroke]

                # Update coordinate lists
                all_x_coords.extend(stroke_x)
                all_y_coords.extend(stroke_y)
                sample_x_coords.extend(stroke_x)
                sample_y_coords.extend(stroke_y)

                # Number of points in this stroke
                num_points = len(stroke)
                num_points_per_stroke.append(num_points)
                total_points += num_points

                # Count floating-point coordinates
                for x, y in zip(stroke_x, stroke_y):
                    total_coords += 2
                    if not x.is_integer():
                        num_float_coords += 1
                    if not y.is_integer():
                        num_float_coords += 1

                # Calculate the distance between consecutive points in the stroke
                for i in range(1, len(stroke)):
                    point1 = stroke[i - 1]
                    point2 = stroke[i]
                    dist = distance.euclidean(point1, point2)
                    if dist < 100:  # Only include distances less than 100
                        distances.append(dist)

            if total_points < 2:
                print(f"Skipping sample with less than 2 points: {sample_id}")

            # Calculate bounding box size for this sample
            if sample_x_coords and sample_y_coords:
                min_x_sample = min(sample_x_coords)
                max_x_sample = max(sample_x_coords)
                min_y_sample = min(sample_y_coords)
                max_y_sample = max(sample_y_coords)
                width = max_x_sample - min_x_sample
                height = max_y_sample - min_y_sample
                bounding_box_sizes.append((width, height))
                bbox_area = width * height

                # Track samples with min/max points and bounding box sizes
                if total_points < min_points:
                    min_points = total_points
                    min_points_sample = strokes
                    dataset_min_points_sample = strokes
                if total_points > max_points:
                    max_points = total_points
                    max_points_sample = strokes
                    dataset_max_points_sample = strokes
                if bbox_area < min_bbox_area:
                    min_bbox_area = bbox_area
                    min_bbox_sample = strokes
                    dataset_min_bbox_sample = strokes
                if bbox_area > max_bbox_area:
                    max_bbox_area = bbox_area
                    max_bbox_sample = strokes
                    dataset_max_bbox_sample = strokes
                if width < min_width:
                    min_width = width
                    narrowest_sample = strokes
                    dataset_narrowest_sample = strokes
                if width > max_width:
                    max_width = width
                    widest_sample = strokes
                    dataset_widest_sample = strokes
                if height < min_height:
                    min_height = height
                    shortest_sample = strokes
                    dataset_shortest_sample = strokes
                    print(f"Min bbox sample: {sample_id}")
                if height > max_height:
                    max_height = height
                    tallest_sample = strokes
                    dataset_tallest_sample = strokes
                max_dim = max(width, height)
                if max_dim < min_max_dim:
                    min_max_dim = max_dim
                    smallest_max_dim_sample = strokes
                    dataset_smallest_max_dim_sample = strokes

            # Track number of coordinates per sample
            num_coordinates = len(sample_x_coords)
            num_coordinates_per_sample.append(num_coordinates)
            if num_coordinates < min_coordinates:
                min_coordinates = num_coordinates
            if num_coordinates > max_coordinates:
                max_coordinates = num_coordinates

    # Convert coordinate lists to numpy arrays
    all_x_coords = np.array(all_x_coords)
    all_y_coords = np.array(all_y_coords)

    # Calculate min and max number of strokes per symbol
    if num_points_per_stroke:
        min_strokes_per_symbol = min(num_points_per_stroke)
        max_strokes_per_symbol = max(num_points_per_stroke)
    else:
        min_strokes_per_symbol = 0
        max_strokes_per_symbol = 0

    # Calculate min, max, and average bounding box areas
    areas = [w * h for w, h in bounding_box_sizes]
    if areas:
        avg_bbox_area = np.mean(areas)
    else:
        avg_bbox_area = 0

    # Calculate min and max coordinates
    min_x = np.min(all_x_coords)
    max_x = np.max(all_x_coords)
    min_y = np.min(all_y_coords)
    max_y = np.max(all_y_coords)

    # Calculate percentage of floating-point coordinates
    percent_float_coords = (num_float_coords / total_coords) * 100 if total_coords > 0 else 0

    # Print statistics
    print("\n--- Statistics ---")
    print(f"Min strokes per symbol: {min_strokes_per_symbol}")
    print(f"Max strokes per symbol: {max_strokes_per_symbol}")
    print(f"Min X: {min_x}")
    print(f"Max X: {max_x}")
    print(f"Min Y: {min_y}")
    print(f"Max Y: {max_y}")
    print(f"Total coordinates: {total_coords}")
    print(f"Number of floating-point coordinates: {num_float_coords}")
    print(f"Percentage of floating-point coordinates: {percent_float_coords:.2f}%")
    print(f"Min coordinates per sample: {min_coordinates}")
    print(f"Max coordinates per sample: {max_coordinates}")
    print(f"Min bounding box area: {min_bbox_area:.2f}")
    print(f"Max bounding box area: {max_bbox_area:.2f}")
    print(f"Average bounding box area: {avg_bbox_area:.2f}")
    print(f"Narrowest width: {min_width}")
    print(f"Widest width: {max_width}")
    print(f"Shortest height: {min_height}")
    print(f"Tallest height: {max_height}")
    print(f"Smallest max(width, height): {min_max_dim}")

    # Prepare plots for entire dataset, if no specific key was provided
    # Prepare bounding box sizes for histogram
    widths = [size[0] for size in bounding_box_sizes]
    heights = [size[1] for size in bounding_box_sizes]
    max_of_wh = [max(w, h) for w, h in bounding_box_sizes]

    # Plot heatmap of coordinate points
    plt.figure(figsize=(8, 6))
    plt.hist2d(all_x_coords, all_y_coords, bins=100, cmap="hot")
    plt.colorbar(label="Number of Points")
    plt.title("Heatmap of Coordinate Points for All Symbols")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Invert y-axis to match drawing coordinates
    plt.show()

    # Plot histogram of bounding box areas
    plt.figure(figsize=(8, 6))
    plt.hist(areas, bins=30, color="blue", edgecolor="black")
    plt.title("Histogram of Bounding Box Areas for All Symbols")
    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency")
    plt.show()

    # Plot histogram of bounding box widths
    plt.figure(figsize=(8, 6))
    plt.hist(widths, bins=30, color="green", edgecolor="black")
    plt.title("Histogram of Bounding Box Widths for All Symbols")
    plt.xlabel("Width")
    plt.ylabel("Frequency")
    plt.show()

    # Plot histogram of bounding box heights
    plt.figure(figsize=(8, 6))
    plt.hist(heights, bins=30, color="purple", edgecolor="black")
    plt.title("Histogram of Bounding Box Heights for All Symbols")
    plt.xlabel("Height")
    plt.ylabel("Frequency")
    plt.show()

    # Plot histogram of maximum of width and height
    plt.figure(figsize=(8, 6))
    plt.hist(max_of_wh, bins=30, color="brown", edgecolor="black")
    plt.title("Histogram of Maximum of Width and Height for All Symbols")
    plt.xlabel("Max(Width, Height)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot histogram of distances between consecutive points (less than 100)
    plt.figure(figsize=(8, 6))
    plt.hist(distances, bins=100, color="orange", edgecolor="black")
    plt.title("Histogram of Distances Between Consecutive Points for All Symbols (Distance < 100)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()

    # Plot histogram of number of coordinates per sample
    plt.figure(figsize=(8, 6))
    plt.hist(num_coordinates_per_sample, bins=30, color="red", edgecolor="black")
    plt.title("Histogram of Number of Coordinates per Sample for All Symbols")
    plt.xlabel("Number of Coordinates")
    plt.ylabel("Frequency")
    plt.show()

    # Plot histogram of symbol frequencies
    # cursor.execute("SELECT key, COUNT(*) FROM samples GROUP BY key ORDER BY COUNT(*) ASC")
    # symbol_frequencies = cursor.fetchall()
    # keys = [row[0] for row in symbol_frequencies]
    # frequencies = [row[1] for row in symbol_frequencies]
    # median_frequency = np.median(frequencies)
    #
    # plt.figure(figsize=(12, 6))
    # plt.bar(keys, frequencies, color="cyan", edgecolor="black")
    # plt.title(f"Histogram of Symbol Frequencies (Median Frequency: {median_frequency})")
    # plt.xlabel("Symbol Key")
    # plt.ylabel("Frequency")
    # plt.xticks(rotation=90, fontsize=8)
    # plt.tight_layout()
    # plt.show()

    # Plot additional visualizations for specific samples, or for dataset-level extremes
    def plot_strokes(_strokes, title):
        fig, axes = plt.subplots(2, 1, figsize=(5, 10))
        fig.suptitle(title)
        # Plot original strokes with points
        axes[0].set_title("With Points")
        for _stroke in _strokes:
            _stroke_x = [point[0] for point in _stroke]
            _stroke_y = [point[1] for point in _stroke]
            axes[0].plot(_stroke_x, _stroke_y, marker="o")

        # Plot original _strokes without points
        for _stroke in _strokes:
            _stroke_x = [point[0] for point in _stroke]
            _stroke_y = [point[1] for point in _stroke]
            axes[1].plot(_stroke_x, _stroke_y)

        for _i in range(2):
            axes[_i].axis("equal")
            axes[_i].set_xlabel("X Coordinate")
            axes[_i].set_ylabel("Y Coordinate")
            axes[_i].set_xlim(0, 1000)
            axes[_i].set_ylim(1000, 0)
            axes[_i].set_aspect("equal", adjustable="box")
        plt.show()

    if symbol_key is not None:
        if isinstance(symbol_key, list) or isinstance(symbol_key, tuple):
            symbol_key = symbol_key[0] + "..."
        if min_points_sample:
            plot_strokes(min_points_sample, f'Strokes with Least Points for Symbol "{symbol_key}"')
        if max_points_sample:
            plot_strokes(max_points_sample, f'Strokes with Most Points for Symbol "{symbol_key}"')
        if min_bbox_sample:
            plot_strokes(
                min_bbox_sample, f'Strokes with Smallest Bounding Box for Symbol "{symbol_key}"'
            )
        if max_bbox_sample:
            plot_strokes(
                max_bbox_sample, f'Strokes with Largest Bounding Box for Symbol "{symbol_key}"'
            )
        if narrowest_sample:
            plot_strokes(narrowest_sample, f'Narrowest Sample for Symbol "{symbol_key}"')
        if widest_sample:
            plot_strokes(widest_sample, f'Widest Sample for Symbol "{symbol_key}"')
        if shortest_sample:
            plot_strokes(shortest_sample, f'Shortest Sample for Symbol "{symbol_key}"')
        if tallest_sample:
            plot_strokes(tallest_sample, f'Tallest Sample for Symbol "{symbol_key}"')
        if smallest_max_dim_sample:
            plot_strokes(
                smallest_max_dim_sample,
                f'Sample with Smallest Max(Width, Height) for Symbol "{symbol_key}"',
            )
    else:
        if dataset_min_points_sample:
            plot_strokes(dataset_min_points_sample, "Strokes with Least Points for Entire Dataset")
        if dataset_max_points_sample:
            plot_strokes(dataset_max_points_sample, "Strokes with Most Points for Entire Dataset")
        if dataset_min_bbox_sample:
            plot_strokes(
                dataset_min_bbox_sample, "Strokes with Smallest Bounding Box for Entire Dataset"
            )
        if dataset_max_bbox_sample:
            plot_strokes(
                dataset_max_bbox_sample, "Strokes with Largest Bounding Box for Entire Dataset"
            )
        if dataset_narrowest_sample:
            plot_strokes(dataset_narrowest_sample, "Narrowest Sample for Entire Dataset")
        if dataset_widest_sample:
            plot_strokes(dataset_widest_sample, "Widest Sample for Entire Dataset")
        if dataset_shortest_sample:
            plot_strokes(dataset_shortest_sample, "Shortest Sample for Entire Dataset")
        if dataset_tallest_sample:
            plot_strokes(dataset_tallest_sample, "Tallest Sample for Entire Dataset")
        if dataset_smallest_max_dim_sample:
            plot_strokes(
                dataset_smallest_max_dim_sample,
                "Sample with Smallest Max(Width, Height) for Entire Dataset",
            )

    conn.close()


if __name__ == "__main__":
    db_path = "../database/handtex.db"

    # Check if symbol key is provided as a command-line argument
    if len(sys.argv) < 2:
        symbol_key = None
    else:
        symbol_key = sys.argv[1]

    import handtex.symbol_relations as sr

    symbol_data = sr.SymbolData()
    symbol_key = symbol_data.get_similarity_group("latex2e-OT1-/")
    # symbol_key = "latex2e-OT1-_backslash"

    analyze_symbol_data(db_path, symbol_key)
