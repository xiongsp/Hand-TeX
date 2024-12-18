import csv
import json
import random
import sqlite3
from functools import cache
from itertools import cycle
from importlib import resources
from math import ceil

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
import handtex.sketchpad as sp


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
    real_data_count: int, max_factor: float = 10, min_factor: float = 0.2
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


class StrokeDataset(Dataset):
    def __init__(
        self,
        db_path,
        symbol_data: sr.SymbolData,
        image_size: int,
        label_encoder: LabelEncoder,
        random_seed: int,
        validation_split: float = 0.1,
        train: bool = True,
        shuffle: bool = True,
        random_augmentation: bool = True,
        stroke_cache: dict[str, list[list[tuple[int, int]]]] = None,
        debug_single_sample_only: bool = False,
    ):
        """
        The primary keys list consists of tuples containing the following:
        - int: The primary key of the sample in the database.
        - list[Transformation]: List of transformations to apply to the strokes before using it.
          These are a result of using symmetries to augment the data.
        - tuple[Negation | None, int]: An optional negation and the id of the slash to apply to the symbol.
        - int | None: If not None, this is the seed to use for random augmentation.
          It is imperative that this be stored, so that the training and validation datasets
          generate the same pool of data and thus can be split consistently.


        :param db_path: Path to the SQLite database.
        :param symbol_data: SymbolData object containing symbol metadata.
        :param image_size: Size of the generated images (images are square)
        :param label_encoder: LabelEncoder object to encode labels.
        :param random_seed: Seed for the random number generator. Generator for training and validation MUST get the same.
        :param validation_split: Fraction of the data to use for validation.
        :param train: If True, load training data, else load validation data.
        :param shuffle: If True, shuffle the data before making the training/validation split.
        :param random_augmentation: If True, augment the data with random transformations.
        :param stroke_cache: Cache of stroke data for each symbol key, alternative to loading from database.
        :param debug_single_sample_only: If True, only load a single sample for debugging.
        """
        self.db_path = db_path
        self.image_size = image_size
        self.primary_keys: list[
            tuple[
                int,
                tuple[st.Transformation, ...],
                tuple[st.Negation | None, int],
                int | None,
            ]
        ] = []
        self.symbol_keys = []
        self.train = train
        self.stroke_cache = stroke_cache

        random.seed(random_seed)

        # Load primary keys and symbol keys from the database for the provided symbol keys
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        @cache
        def load_primary_keys(_symbol_keys: str | tuple[str]) -> list[tuple[int]]:
            if isinstance(_symbol_keys, str):
                _symbol_keys = (_symbol_keys,)
            command = f"SELECT id FROM samples WHERE key IN ({','.join(['?']*len(_symbol_keys))})"
            if debug_single_sample_only:
                command += " LIMIT 1"
            cursor.execute(
                command,
                _symbol_keys,
            )
            return cursor.fetchall()

        # For negations, we need a cycle of the symbol keys to use for the slash.
        vertical_line_keys = symbol_data.get_similarity_group("latex2e-OT1-|")
        negation_cycle = cycle(load_primary_keys(vertical_line_keys))

        for symbol_key in symbol_data.leaders:

            samples: list[
                tuple[
                    int,
                    tuple[st.Transformation, ...],
                    tuple[st.Negation | None, int],
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

            for current_key, transformations, negation in symbol_data.all_paths_to_symbol(
                symbol_key
            ):
                # The [0] access is to unwrap the sqlite row tuples.
                samples.extend(
                    (row[0], transformations, (negation, slash_id[0]), None)
                    for row, slash_id in zip(load_primary_keys(current_key), negation_cycle)
                )
                if current_key in symbol_data.get_similarity_group(symbol_key) and transformations:
                    # If we don't have transformations, those are just similarity taking over
                    # with the identity transform, that counts as real data.
                    # So here, we do have a self-symmetry applied from its similarity group.
                    self_symmetry_count += len(load_primary_keys(current_key))

                if current_key not in symbol_data.get_similarity_group(symbol_key):
                    other_symmetry_count += len(load_primary_keys(current_key))

                if negation is not None:
                    negation_count += len(load_primary_keys(current_key))

            assert samples, f"No samples found for symbol key: {symbol_key}"
            # Augment the data to balance the classes.
            if random_augmentation:
                augmentation_count = augmentation_amount(real_data_count)
                for _ in range(augmentation_count):
                    symbol, transformations, negation_tuple, _ = random.choice(samples)
                    samples.append(
                        (symbol, transformations, negation_tuple, random.randint(0, 2**32 - 1))
                    )

            # Shuffle the rows to get more variety in drawings, since drawings
            # by the same person are sequentially stored in the database.
            if shuffle:
                random.shuffle(samples)

            split_idx = ceil(len(samples) * validation_split)
            if train:
                selected_rows = samples[split_idx:]
                print(
                    f"Loaded {len(samples)} total samples of {symbol_key}, "
                    f"with {real_data_count} real data, "
                    f"{self_symmetry_count} self-symmetries, "
                    f"{other_symmetry_count} other symmetries, "
                    f"and {augmentation_count} random augmentations. "
                    f"Reserving {len(selected_rows)} for training. "
                    f"Reserving {split_idx} for validation."
                )
            else:
                selected_rows = samples[:split_idx]

            self.primary_keys.extend(selected_rows)
            self.symbol_keys.extend([symbol_key] * len(selected_rows))

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

    def __len__(self):
        return len(self.primary_keys)

    def range_for_symbol(self, symbol: str) -> range:
        """
        Get the range of indices for a given symbol.
        """
        start = self.symbol_keys.index(symbol)
        end = len(self.symbol_keys) - self.symbol_keys[::-1].index(symbol)
        return range(start, end)

    def load_transformed_strokes(self, idx):
        primary_key, required_transforms, (negation, slash_id), random_augmentation_seed = (
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

        # Augment the data with a random transformation.
        # The transformation is applied to the strokes before converting them to an image.
        if random_augmentation_seed is not None:
            random.seed(random_augmentation_seed)
            operation = random.randint(0, 2)
            if operation == 0:
                trans_mats.append(ig.rotation_matrix(np.random.uniform(-5, 5)))
            elif operation == 1:
                trans_mats.append(
                    ig.scale_matrix(np.random.uniform(0.9, 1), np.random.uniform(0.9, 1))
                )
            elif operation == 2:
                trans_mats.append(
                    ig.skew_matrix(np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
                )

        # Apply the transformations to the stroke data.
        if trans_mats:
            stroke_data = ig.apply_transformations(stroke_data, trans_mats)

        # Next, prepare the negation, if any.
        if negation is not None:
            negation_stroke_data = self.load_stroke_data(slash_id)
            trans_mats = [
                ig.scale_matrix(negation.scale_factor, negation.scale_factor),
                ig.rotation_matrix(negation.vert_angle),
                ig.translation_matrix(-negation.x_offset, -negation.y_offset, 1000),
            ]
            negation_stroke_data = ig.apply_transformations(negation_stroke_data, trans_mats)
            stroke_data += negation_stroke_data
            # # Now we may need to rescale the image to fit the strokes, so apply an identity transform
            # # and let the transformation function handle the scaling.
            # stroke_data = ig.apply_transformations(stroke_data, np.eye(3))
            stroke_data, _, _, _ = sp.rescale_and_center_viewport(stroke_data, 1000, 1000)

        symbol_key = self.symbol_keys[idx]

        return stroke_data, symbol_key

    def __getitem__(self, idx):
        stroke_data, _ = self.load_transformed_strokes(idx)

        img = ig.strokes_to_grayscale_image_cv2(stroke_data, self.image_size)

        # Apply the transform to convert the image to a tensor and normalize it
        img_tensor = self.transform(img)
        label_tensor = torch.tensor(
            self.encoded_labels[idx], dtype=torch.long
        )  # Convert label to tensor
        return img_tensor, label_tensor  # Return image tensor and label

    def load_stroke_data(self, primary_key):
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
    with ut.resource_path(training.database, "handtex.db") as path:
        database_path = path

    with resources.path(handtex.data.symbol_metadata, "symbol_frequency.csv") as path:
        frequencies_path = path

    with resources.path(handtex.data.symbol_metadata, "augmented_symbol_frequency.csv") as path:
        augmented_frequencies_path = path

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
    image_size = 56

    # Create training and validation datasets and dataloaders
    all_samples = StrokeDataset(
        db_path,
        symbol_data,
        image_size,
        label_encoder,
        random_seed=0,
        validation_split=0,
        train=True,
        shuffle=False,
        random_augmentation=False,
        debug_single_sample_only=True,
    )

    # Show all the samples for a given symbol.
    symbol = "stix-OT1-_lbag"
    assert (
        symbol in symbol_data.symbol_keys
    ), f"Symbol '{symbol}' not found in the dataset or not a leader"
    symbol_strokes = (
        all_samples.load_transformed_strokes(idx) for idx in all_samples.range_for_symbol(symbol)
    )
    for idx, ((strokes, symbol), (current_key, transformations, negation)) in enumerate(
        zip(symbol_strokes, symbol_data.all_paths_to_symbol(symbol))
    ):
        img = ig.strokes_to_grayscale_image_cv2(strokes, image_size)
        print(
            f"Current symbol: {current_key} with transformations: {transformations} and negation: {negation}"
        )
        cv2.imshow(f"{symbol} {idx}", img)
        # wait for a key press
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
