import csv
import json
import random
import sqlite3
from enum import Enum
from functools import cache
from itertools import cycle
from importlib import resources
from math import floor

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

import handtex.data.symbol_metadata
import handtex.structures as st
import handtex.symbol_relations as sr
import handtex.utils as ut
import training.database
import training.image_gen as ig
import training.shape_classifier as sc
import training.hyperparameters as hyp
import handtex.sketchpad as sp


class DataSplit(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


def build_stroke_cache(db_path: str) -> dict[str, list[list[tuple[int, int]]]]:
    """
    Build a cache of the stroke data for each symbol in the database.
    This maps the id to the strokes, the key information is lost.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, strokes FROM samples")
    rows = cursor.fetchall()
    stroke_cache = {key: json.loads(strokes) for key, strokes in rows}

    conn.close()
    return stroke_cache


def augmentation_amount(
    real_data_count: int, max_factor: float = 20, min_factor: float = 0.2
) -> int:
    """
    Calculate the amount of augmented data to generate based on the real data count.
    """
    power_base = 1.2
    stretch = 0.05

    nominator = -2 * (max_factor - min_factor)
    denominator = 1 + power_base ** (-stretch * real_data_count)
    offset = min_factor - nominator
    factor = nominator / denominator + offset

    return int(factor * real_data_count)


def get_data_split(
    data: list, desired_split: DataSplit, split_at_percentages: dict[DataSplit, int]
) -> list:
    n = len(data)

    # Handle cases where we have less than 3 samples.
    if n < 3:
        # There is no point splitting this up.
        return data

    if sum(split_at_percentages.values()) != 100:
        raise ValueError("Split percentages must sum to 100")

    # Sort the splits in the order defined by the enum.
    splits = sorted(split_at_percentages.keys(), key=lambda s: s.value)

    # Compute the ideal (raw) count for each split.
    # We use floor, but guarantee that each split gets at least one sample.
    counts = {}
    fractions = {}
    for s in splits:
        raw = n * split_at_percentages[s] / 100
        floor_val = floor(raw)
        # Guarantee at least one sample per split
        if floor_val < 1:
            floor_val = 1
            fraction = 0
        else:
            fraction = raw - floor_val
        counts[s] = floor_val
        fractions[s] = fraction

    # Compute how many samples are still unassigned.
    current_sum = sum(counts.values())
    leftover = n - current_sum

    # Distribute any leftover samples, one-by-one, starting with the split
    # that had the highest fractional part.
    for s in sorted(splits, key=lambda s: fractions[s], reverse=True):
        if leftover <= 0:
            break
        counts[s] += 1
        leftover -= 1

    # Now create contiguous, disjoint segments based on these counts.
    boundaries = {}
    start = 0
    for s in splits:
        count = counts[s]
        boundaries[s] = (start, start + count)
        start += count

    start_idx, end_idx = boundaries[desired_split]
    return data[start_idx:end_idx]


class StrokeDataset(Dataset):
    def __init__(
        self,
        db_path,
        symbol_data: sr.SymbolData,
        image_size: int,
        label_encoder: LabelEncoder,
        random_seed: int,
        split_percentages: dict[DataSplit, int],
        split: DataSplit = DataSplit.TRAIN,
        class_limit: int | None = None,
        random_augmentation: bool = True,
        stroke_cache: dict[str, list[list[tuple[int, int]]]] = None,
        debug_single_sample_only: bool = False,
        distribution_stats: dict[str, tuple[int, int, int, int, int, int]] | None = None,
    ):
        """
        The primary keys list consists of tuples containing the following:
        - int: The primary key of the sample in the database.
        - list[Transformation]: List of transformations to apply to the strokes before using it.
          These are a result of using symmetries to augment the data.
        - tuple[Negation | Inside | None, int]: An optional negation and the id of the slash / shape to apply to the symbol.
        - int | None: If not None, this is the seed to use for random augmentation.
          It is imperative that this be stored, so that the training and validation datasets
          generate the same pool of data and thus can be split consistently.


        :param db_path: Path to the SQLite database.
        :param symbol_data: SymbolData object containing symbol metadata.
        :param image_size: Size of the generated images (images are square)
        :param label_encoder: LabelEncoder object to encode labels.
        :param random_seed: Seed for the random number generator. Generator for training and validation MUST get the same.
        :param split: DataSplit enum to determine which split to use.
        :param random_augmentation: If True, augment the data with random transformations.
        :param stroke_cache: Cache of stroke data for each symbol key, alternative to loading from database.
        :param debug_single_sample_only: If True, only load a single sample for debugging.
        :param distribution_stats: If not none, it is populated with the statistics of the dataset.
        """
        self.db_path = db_path
        self.image_size = image_size
        self.primary_keys: list[
            tuple[
                int,
                tuple[st.Transformation, ...],
                tuple[st.Negation | st.Inside | None, int],
                int | None,
            ]
        ] = []
        self.symbol_keys = []
        self.split = split
        self.stroke_cache = stroke_cache

        # Store the seed and create an instance-level RNG so that bootstrapping is isolated.
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

        # Load primary keys and symbol keys from the database for the provided symbol keys
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        @cache
        def load_primary_keys(
            _symbol_keys: str | tuple[str] | list[str],
            *,
            ignore_single_debug_sample_limit: bool = False,
        ) -> list[int]:
            nonlocal cursor, debug_single_sample_only

            if isinstance(_symbol_keys, str):
                _symbol_keys = (_symbol_keys,)
            command = f"SELECT id FROM samples WHERE key IN ({','.join(['?']*len(_symbol_keys))})"
            if debug_single_sample_only and not ignore_single_debug_sample_limit:
                command += " LIMIT 1"
            cursor.execute(
                command,
                _symbol_keys,
            )
            # Unwrap the tuples, since squealite returns each row as such, even when selecting a single column.
            cursor_samples = [row[0] for row in cursor.fetchall()]

            # If there is only one sample, no shuffling is possible.
            if len(cursor_samples) == 1:
                return cursor_samples

            # Bootstrapping: shuffle deterministically.
            # Derive a seed from the global random_seed and the _symbol_keys
            # (this ensures that the same symbol group always shuffles identically)
            seed_for_shuffle = hash((self.random_seed, tuple(_symbol_keys)))
            local_rng = random.Random(seed_for_shuffle)
            local_rng.shuffle(cursor_samples)

            nonlocal split
            return get_data_split(cursor_samples, split, split_percentages)

        # For negations, we need a cycle of the symbol keys to use for the slash.
        vertical_line_keys = symbol_data.get_similarity_group("latex2e-|")
        negation_cycle = cycle(load_primary_keys(vertical_line_keys))

        # For inside-relations, we need a cycle for circles, squares, and triangles.
        # These need to be filtered for fitness to fit around a symbol.
        circle_keys = symbol_data.get_similarity_group("latex2e-_bigcirc")
        square_keys = symbol_data.get_similarity_group("amssymb-_square")
        triangle_keys = symbol_data.get_similarity_group("latex2e-_bigtriangleup")
        circle_cycle = cycle(
            filter(
                lambda s_id: sc.is_good_circle(self.load_stroke_data(s_id))[0],
                load_primary_keys(circle_keys, ignore_single_debug_sample_limit=True),
            )
        )
        square_cycle = cycle(
            filter(
                lambda s_id: sc.is_good_square(self.load_stroke_data(s_id))[0],
                load_primary_keys(square_keys, ignore_single_debug_sample_limit=True),
            )
        )
        triangle_cycle = cycle(
            filter(
                lambda s_id: sc.is_good_triangle(self.load_stroke_data(s_id))[0],
                load_primary_keys(triangle_keys, ignore_single_debug_sample_limit=True),
            )
        )

        for symbol_key in symbol_data.leaders:

            samples: list[
                tuple[
                    int,
                    tuple[st.Transformation, ...],
                    tuple[st.Negation | st.Inside | None, int],
                    int | None,
                ]
            ] = []

            # Identical values to augmented_symbol_frequency.csv
            real_data_count = sum(
                len(load_primary_keys(ancestor))
                for ancestor in symbol_data.all_symbols_to_symbol(symbol_key)
            )
            self_symmetry_count = 0
            other_symmetry_count = 0
            negation_count = 0
            augmentation_count = 0
            inside_count = 0

            for current_key, transformations, composition in symbol_data.all_paths_to_symbol(
                symbol_key
            ):
                for symbol_id in load_primary_keys(current_key):
                    # We need to check what composition we have, if any.
                    if composition is None:
                        composition_id = 0
                    elif isinstance(composition, st.Negation):
                        composition_id = next(negation_cycle)
                    elif composition == st.Inside.Circle:
                        composition_id = next(circle_cycle)
                    elif composition == st.Inside.Square:
                        composition_id = next(square_cycle)
                    elif composition == st.Inside.Triangle:
                        composition_id = next(triangle_cycle)
                    else:
                        raise ValueError(f"Unknown composition: {composition}")

                    samples.append(
                        (symbol_id, transformations, (composition, composition_id), None)
                    )

                if current_key in symbol_data.get_similarity_group(symbol_key) and transformations:
                    # If we don't have transformations, those are just similarity taking over
                    # with the identity transform, that counts as real data.
                    # So here, we do have a self-symmetry applied from its similarity group.
                    self_symmetry_count += len(load_primary_keys(current_key))

                if current_key not in symbol_data.get_similarity_group(symbol_key):
                    other_symmetry_count += len(load_primary_keys(current_key))

                if isinstance(composition, st.Negation):
                    negation_count += len(load_primary_keys(current_key))

                if isinstance(composition, st.Inside):
                    inside_count += len(load_primary_keys(current_key))

            # assert samples, f"No samples found for symbol key: {symbol_key}"

            # Augment the data to balance the classes.
            if random_augmentation:
                augmentation_count = augmentation_amount(real_data_count)
                for _ in range(augmentation_count):
                    symbol, transformations, composition_tuple, _ = self.rng.choice(samples)
                    # Use 16 bits of randomness, so as not to overflow the perlin noise algorithm.
                    samples.append(
                        (symbol, transformations, composition_tuple, self.rng.randint(0, 2**16 - 1))
                    )

            if class_limit is not None:
                samples = samples[:class_limit]

            # Only report stats for the training split to cut down on log spam.
            if split.TRAIN:
                print(
                    f"Loaded {len(samples)} total samples of {symbol_key}, "
                    f"with {real_data_count} real data, "
                    f"{self_symmetry_count} self-symmetries, "
                    f"{other_symmetry_count} other symmetries, "
                    f"{negation_count} negations, "
                    f"{inside_count} inside relations,"
                    f"and {augmentation_count} random augmentations. "
                )
                if distribution_stats is not None:
                    distribution_stats[symbol_key] = (
                        len(samples),
                        real_data_count,
                        self_symmetry_count,
                        other_symmetry_count,
                        negation_count,
                        augmentation_count,
                    )

            self.primary_keys.extend(samples)
            self.symbol_keys.extend([symbol_key] * len(samples))

        conn.close()

        # Encode labels into integers
        self.encoded_labels = label_encoder.transform(self.symbol_keys)

        # Define a transform to convert the images to tensors and normalize them
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts the image to a PyTorch tensor (C, H, W) with values in [0, 1]
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to range [-1, 1]
            ]
        )

    def __len__(self) -> int:
        return len(self.primary_keys)

    def range_for_symbol(self, symbol: str) -> range:
        """
        Get the range of indices for a given symbol.
        """
        start = self.symbol_keys.index(symbol)
        end = len(self.symbol_keys) - self.symbol_keys[::-1].index(symbol)
        return range(start, end)

    def load_transformed_strokes(self, idx) -> tuple[list[list[tuple[int, int]]], str]:
        primary_key, required_transforms, (composition, composite_id), random_augmentation_seed = (
            self.primary_keys[idx]
        )
        stroke_data = self.load_stroke_data(primary_key)
        # If a symmetric character was used, we will need to apply it's transformation.
        # We may have multiple options here.
        trans_mats = []
        for transformation in required_transforms:
            if transformation.is_rotation:
                trans_mats.append(ig.rotation_matrix(transformation.angle))
            else:
                trans_mats.append(ig.reflection_matrix(transformation.angle))

        # Apply the transformations to the stroke data.
        if trans_mats:
            stroke_data = ig.apply_transformations(stroke_data, trans_mats)

        # Rescale the image to ensure the rotations and reflections fit within the image bounds.
        # This is necessary to ensure the compositions align correctly.
        stroke_data, _, _, _ = sp.rescale_and_center_viewport(stroke_data, 1000, 1000)

        # Next, prepare the negation, if any.
        if isinstance(composition, st.Negation):
            negation: st.Negation = composition
            slash_id: int = composite_id

            negation_stroke_data = self.load_stroke_data(slash_id)
            # Use -y_offset, since the negation views up as positive y, but in the image it's
            # in the negative y direction.
            trans_mats = [
                ig.scale_matrix(negation.scale_factor, negation.scale_factor),
                ig.rotation_matrix(negation.vert_angle),
                ig.translation_matrix(negation.x_offset, -negation.y_offset, 1000),
            ]
            negation_stroke_data = ig.apply_transformations(negation_stroke_data, trans_mats)
            stroke_data += negation_stroke_data
        # Otherwise, prepare the inside relation, if any.
        elif isinstance(composition, st.Inside):
            inside: st.Inside = composition
            outer_id: int = composite_id

            outer_stroke_data = self.load_stroke_data(outer_id)
            # Find out how large we need to scale the outer symbol to fit the inner symbol.
            center_x, center_y = 500, 500
            square_side_length = 1000
            if inside == st.Inside.Circle:
                _, square_side_length = sc.is_good_circle(outer_stroke_data)
            elif inside == st.Inside.Square:
                _, square_side_length = sc.is_good_square(outer_stroke_data)
            elif inside == st.Inside.Triangle:
                _, square_side_length, center_x, center_y = sc.is_good_triangle(outer_stroke_data)
            else:
                raise ValueError(f"Unknown inside relation: {inside}")

            scale = square_side_length / 1000
            scale *= 0.9  # Add a 10% margin to fit.

            trans_mats = [
                ig.scale_matrix(scale, scale),
                ig.translation_matrix((center_x - 500) / 1000, (center_y - 500) / 1000, 1000),
            ]
            stroke_data = ig.apply_transformations(stroke_data, trans_mats)
            stroke_data += outer_stroke_data

        # Augment the data with a random transformation.
        # The transformation is applied to the strokes before converting them to an image.
        trans_mats = []
        if random_augmentation_seed is not None:
            # Apply some type of random augmentation, using perlin noise 7/10 times.
            local_rng = random.Random(random_augmentation_seed)
            operation = local_rng.randint(0, 9)
            numpy_rng = np.random.RandomState(random_augmentation_seed)

            if operation == 0:
                trans_mats.append(ig.rotation_matrix(numpy_rng.uniform(-5, 5)))
            elif operation == 1:
                trans_mats.append(
                    ig.scale_matrix(numpy_rng.uniform(0.9, 1), numpy_rng.uniform(0.9, 1))
                )
            elif operation == 2:
                trans_mats.append(
                    ig.skew_matrix(numpy_rng.uniform(-0.1, 0.1), numpy_rng.uniform(-0.1, 0.1))
                )
            else:
                stroke_data = ig.augment_strokes_with_perlin(
                    stroke_data, seed=random_augmentation_seed
                )

        # Apply the transformations to the stroke data.
        if trans_mats:
            stroke_data = ig.apply_transformations(stroke_data, trans_mats)

        # Rescale the image to ensure the rotations and reflections fit within the image bounds.
        stroke_data, _, _, _ = sp.rescale_and_center_viewport(stroke_data, 1000, 1000)

        symbol_key = self.symbol_keys[idx]

        return stroke_data, symbol_key

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        stroke_data, _ = self.load_transformed_strokes(idx)

        img = ig.strokes_to_grayscale_image_cv2(stroke_data, self.image_size)

        # Apply the transform to convert the image to a tensor and normalize it
        img_tensor = self.transform(img)
        label_tensor = torch.tensor(
            self.encoded_labels[idx], dtype=torch.long
        )  # Convert label to tensor
        return img_tensor, label_tensor  # Return image tensor and label

    def load_stroke_data(self, primary_key) -> list[list[tuple[int, int]]]:
        if self.stroke_cache is not None:
            return self.stroke_cache[primary_key]
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query the stroke data for the given primary key
        cursor.execute("SELECT strokes FROM samples WHERE id = ?", (primary_key,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise ValueError(f"No stroke data found for primary key: {primary_key}")

        # Load the stroke data from JSON format
        stroke_data = json.loads(row[0])
        return stroke_data


def recalculate_frequencies():
    symbol_data = sr.SymbolData()
    # Limit the number of classes to classify.
    leader_keys = symbol_data.leaders

    # database_path = "database/handtex.db"
    database_path = ut.resource_path(training.database, "handtex.db")

    frequencies_path = ut.resource_path(handtex.data.symbol_metadata, "symbol_frequency.csv")

    augmented_frequencies_path = ut.resource_path(
        handtex.data.symbol_metadata, "augmented_symbol_frequency.csv"
    )

    # Get the frequencies from the database.
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    # cursor.execute("SELECT key, COUNT(*) FROM samples GROUP BY key ORDER BY count ASC")
    # Sort by the count of each symbol in descending order.
    cursor.execute("SELECT key, COUNT(*) AS count FROM samples GROUP BY key ORDER BY count DESC")
    rows = cursor.fetchall()
    frequencies = {key: count for key, count in rows}
    conn.close()

    # Look for symbols that weren't included.
    missing_symbols = set(symbol_data.symbol_keys) - set(frequencies.keys())
    if missing_symbols:
        print(f"Missing frequencies for symbols:")
        for symbol in missing_symbols:
            print(symbol)
        frequencies.update({key: 0 for key in missing_symbols})

    with open(frequencies_path, "w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(frequencies.items())

    # Calculate new augmented frequencies.
    # Sum up the frequencies of all symbols that are it's ancestor as well.
    augmented_frequencies = {}
    for leader in leader_keys:
        augmented_frequencies[leader] = sum(
            frequencies[ancestor] for ancestor in symbol_data.all_symbols_to_symbol(leader)
        )
    # Add in all non-leaders, copying the leader's augmented frequency.
    for key in frequencies:
        if key not in leader_keys:
            augmented_frequencies[key] = augmented_frequencies[symbol_data.to_leader[key]]
    # Dump the new frequencies to a CSV file, sorted by frequency.
    # Sort by name, then by frequency.
    sorted_frequencies = sorted(
        augmented_frequencies.items(), key=lambda item: item[0], reverse=False
    )
    sorted_frequencies = sorted(sorted_frequencies, key=lambda item: item[1], reverse=True)
    with open(augmented_frequencies_path, "w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(sorted_frequencies)

    symbol_mean_freq = sum(frequencies.values()) / len(frequencies)
    augmented_mean_freq = sum(augmented_frequencies.values()) / len(augmented_frequencies)
    median_freq = sorted(frequencies.values())[len(frequencies) // 2]
    augmented_median_freq = sorted(augmented_frequencies.values())[len(augmented_frequencies) // 2]
    std_dev_freq = np.std(list(frequencies.values()))
    augmented_std_dev_freq = np.std(list(augmented_frequencies.values()))
    leader_count = len(leader_keys)
    print(
        f"Mean frequency of all symbols: {symbol_mean_freq:.2f}, median: {median_freq}, std dev: {std_dev_freq:.2f}"
    )
    print(
        f"Mean augmented frequency of symbols: {augmented_mean_freq:.2f}, median: {augmented_median_freq}, std dev: {augmented_std_dev_freq:.2f}"
    )
    print(
        f"Total symbols: {len(symbol_data.all_keys)} | Leader count: {leader_count} | Unique leaders: {len(symbol_data.symbols_grouped_by_transitive_symmetry)}"
    )
    return
    # Plot both together in a bar chart.
    # We want to display the frequency as heights without any labels.
    # Just display the sorted list of heights overlayed on each other.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(
        range(len(augmented_frequencies)),
        sorted(augmented_frequencies.values()),
        label="Leader Symbol Frequencies",
    )
    ax.bar(range(len(frequencies)), sorted(frequencies.values()), label="Symbol Frequencies")
    ax.legend()
    plt.show()


def main():
    # Test data loading.
    # symbols = ut.load_symbols()
    # similar_symbols = ut.load_symbol_metadata_similarity()
    # symbol_keys = ut.select_leader_symbols(list(symbols.keys()), similar_symbols)
    # self_symmetries = ut.load_symbol_metadata_self_symmetry()
    # other_symmetries = ut.load_symbol_metadata_other_symmetry()
    symbol_data = sr.SymbolData()

    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_data.symbol_keys)

    db_path = "database/handtex.db"

    # stats = {}
    stats = None

    if stats is not None:
        # Create training and validation datasets and dataloaders
        all_samples = StrokeDataset(
            db_path,
            symbol_data,
            hyp.image_size,
            label_encoder,
            random_seed=0,
            random_augmentation=True,
            debug_single_sample_only=False,
            distribution_stats=stats,
        )
        # Gather stats and visualize them as stacked columns.
        import matplotlib.pyplot as plt

        # distribution_stats[symbol_key] = (
        #     len(samples),
        #     real_data_count,
        #     self_symmetry_count,
        #     other_symmetry_count,
        #     negation_count,
        #     augmentation_count,
        # )

        # Sort stats by total count.
        sorted_stats = sorted(stats.items(), key=lambda item: item[1][0], reverse=True)
        sorted_keys = [key for key, _ in sorted_stats]
        array = np.array([values for _, values in sorted_stats])
        means = np.mean(array, axis=0)
        medians = np.median(array, axis=0)
        std_devs = np.std(array, axis=0)
        min_vals = np.min(array, axis=0)
        max_vals = np.max(array, axis=0)

        def print_categories_int(values):
            print(
                f"Total: {values[0]}, Real: {values[1]}, Self: {values[2]}, Other: {values[3]}, Negation: {values[4]}, Augmentation: {values[5]}"
            )

        def print_categories_float(values):
            print(
                f"Total: {values[0]:.2f}, Real: {values[1]:.2f}, Self: {values[2]:.2f}, Other: {values[3]:.2f}, Negation: {values[4]:.2f}, Augmentation: {values[5]:.2f}"
            )

        print("Means:")
        print_categories_float(means)
        print("Medians:")
        print_categories_int(medians)
        print("Standard Deviations:")
        print_categories_float(std_devs)
        print("Minimums:")
        print_categories_int(min_vals)
        print("Maximums:")
        print_categories_int(max_vals)

        # Plot the data, sorted by the total count.
        # We want to plot columns for each symbol key, so that the column
        # is divided into the categories, skipping the total.
        # We want to plot the categories as stacked columns.
        plt.bar(
            range(len(stats)),
            array[:, 1],
            label="Real",
            color="blue",
        )
        plt.bar(
            range(len(stats)),
            array[:, 2],
            label="Self",
            color="green",
            bottom=array[:, 1],
        )
        plt.bar(
            range(len(stats)),
            array[:, 3],
            label="Other",
            color="red",
            bottom=array[:, 2] + array[:, 1],
        )
        plt.bar(
            range(len(stats)),
            array[:, 4],
            label="Negation",
            color="purple",
            bottom=array[:, 3] + array[:, 2] + array[:, 1],
        )
        plt.bar(
            range(len(stats)),
            array[:, 5],
            label="Augmentation",
            color="orange",
            bottom=array[:, 4] + array[:, 3] + array[:, 2] + array[:, 1],
        )
        plt.legend()
        plt.xticks(range(len(stats)), sorted_keys, rotation=90)
        plt.show()

        return

    # Create training and validation datasets and dataloaders
    all_samples = StrokeDataset(
        db_path,
        symbol_data,
        hyp.image_size,
        label_encoder,
        random_seed=0,
        random_augmentation=True,
        debug_single_sample_only=False,
        distribution_stats=None,
    )

    test1 = StrokeDataset(
        db_path,
        symbol_data,
        hyp.image_size,
        label_encoder,
        random_seed=0,
        split=DataSplit.TEST,
        class_limit=100,
        random_augmentation=True,
        distribution_stats=None,
    )
    test2 = StrokeDataset(
        db_path,
        symbol_data,
        hyp.image_size,
        label_encoder,
        random_seed=0,
        split=DataSplit.TEST,
        class_limit=100,
        random_augmentation=True,
        distribution_stats=None,
    )
    assert len(test1) == len(test2), "Test datasets should have the same length"
    assert (
        test1.primary_keys == test2.primary_keys
    ), "Test datasets should have the same primary keys"

    # Show all the samples for a given symbol.
    symbol = "stix-_triangleplus"
    leader = symbol_data.to_leader.get(symbol, symbol)
    if symbol != leader:
        print(f"Symbol '{symbol}' is not a leader, using leader '{leader}' instead.")
        symbol = leader
    assert symbol in symbol_data.symbol_keys, f"Symbol '{symbol}' not found in the dataset"

    symbol_strokes = list(
        all_samples.load_transformed_strokes(idx) for idx in all_samples.range_for_symbol(symbol)
    )
    for idx, ((strokes, symbol), (current_key, transformations, composition)) in enumerate(
        zip(symbol_strokes, symbol_data.all_paths_to_symbol(symbol))
    ):
        if len(symbol_strokes) < len(symbol_data.all_paths_to_symbol(symbol)):
            # The stored strokes can only be recorded per path if the symbol the path
            # originates from has a sample in the database. As of writing, 1700 symbols
            # have 0 samples, but when augmented with similarity, symmetry, and composition,
            # each symbol does have a sample.
            # This discrepancy causes the de-synchronization between the existing paths
            # and the expected paths, hence producing wrong info here.
            print("THE CLAIMED PATHS MAY NOT BE CORRECT!!")
        img = ig.strokes_to_grayscale_image_cv2(strokes, hyp.image_size)
        print(
            f"Current symbol: {current_key} with transformations: {transformations} and composition: {composition}"
        )
        cv2.imshow(f"{symbol} {idx}", img)
        # wait for a key press
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

    # If we had more samples than paths, those are augmentations for sure.
    # Just show them now.
    if len(symbol_strokes) > len(symbol_data.all_paths_to_symbol(symbol)):
        for idx, (strokes, symbol) in enumerate(
            symbol_strokes[len(symbol_data.all_paths_to_symbol(symbol)) :]
        ):
            img = ig.strokes_to_grayscale_image_cv2(strokes, hyp.image_size)
            print(f"Current symbol: {symbol} with some augmentation.")
            cv2.imshow(f"{symbol} {idx}", img)
            # wait for a key press
            cv2.waitKey(0)
            # close the window
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
