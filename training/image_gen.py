import random
import numpy as np
import csv
from pathlib import Path
import cv2
from math import ceil
from importlib import resources
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sqlite3
import json
from sklearn.preprocessing import LabelEncoder

import handtex.utils as ut
import handtex.structures as st
import handtex.data.model
import handtex.data.symbol_metadata
import training.database




def build_stroke_cache(db_path: str) -> dict[str, list[list[tuple[int, int]]]]:
    """
    Build a cache of the stroke data for each symbol in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, strokes FROM samples")
    rows = cursor.fetchall()
    cache = {key: json.loads(strokes) for key, strokes in rows}

    conn.close()
    return cache


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


# Propagate self symmetries to similar symbols.
class StrokeDataset(Dataset):
    def __init__(
        self,
        db_path,
        symbol_keys: list[str],
        lookalikes: dict[str, tuple[str, ...]],
        self_symmetries: dict[str, list[st.Symmetry]],
        other_symmetries: dict[str, list[tuple[str, list[st.Symmetry]]]],
        image_size: int,
        label_encoder: LabelEncoder,
        validation_split: float = 0.1,
        train: bool = True,
        shuffle: bool = True,
        random_augmentation: bool = True,
        stroke_cache: dict[str, list[list[tuple[int, int]]]] = None,
    ):
        """
        :param db_path: Path to the SQLite database.
        :param symbol_keys: List of symbol keys to load stroke data for.
        :param lookalikes: Dictionary of lookalike characters.
        :param image_size: Size of the generated images (images are square)
        :param label_encoder: LabelEncoder object to encode labels.
        :param validation_split: Fraction of the data to use for validation.
        :param train: If True, load training data, else load validation data.
        :param shuffle: If True, shuffle the data before making the training/validation split.
        :param random_augmentation: If True, augment the data with random transformations.
        :param stroke_cache: Cache of stroke data for each symbol key, alternative to loading from database.
        """
        self.db_path = db_path
        self.image_size = image_size
        self.primary_keys: list[tuple[int, st.Symmetry, bool]] = []
        self.symbol_keys = []
        self.self_symmetries = self_symmetries
        self.other_symmetries = other_symmetries
        self.train = train
        self.stroke_cache = stroke_cache

        # Load primary keys and symbol keys from the database for the provided symbol keys
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for symbol_key in symbol_keys:

            samples: list[tuple[int, st.Symmetry, bool]] = []
            real_data_count = 0
            self_symmetry_count = 0
            other_symmetry_count = 0
            augmentation_count = 0

            key_and_lookalikes = [symbol_key] + list(lookalikes.get(symbol_key, []))
            cursor.execute(
                f"SELECT id FROM samples WHERE key IN ({','.join(['?']*len(key_and_lookalikes))})",
                key_and_lookalikes,
            )
            rows = cursor.fetchall()
            real_data_count = len(rows)

            self_symmetric_samples = []
            for row in rows:
                samples.append((row[0], st.Symmetry.none, False))
                # If it has self-symmetries, some amount of them will be added to the dataset.
                for symmetry in self.self_symmetries.get(symbol_key, []):
                    self_symmetric_samples.append((row[0], symmetry, False))
            # Limit the number of self-symmetries.
            augmentation_limit = augmentation_amount(
                len(self_symmetric_samples), max_factor=1, min_factor=0.1
            )
            samples.extend(random.choices(self_symmetric_samples, k=augmentation_limit))
            self_symmetry_count = augmentation_limit

            # If the symbol has other symmetries, augment using them.
            if symbol_key in self.other_symmetries:
                for other_key, symmetries in self.other_symmetries[symbol_key]:
                    cursor.execute(
                        "SELECT id FROM samples WHERE key = ?",
                        (other_key,),
                    )
                    other_symmetric_samples = []
                    for row in cursor.fetchall():
                        for symmetry in symmetries:
                            other_symmetric_samples.append((row[0], symmetry, False))
                    # Limit the number of other symmetries.
                    augmentation_limit = augmentation_amount(
                        len(other_symmetric_samples), max_factor=1, min_factor=0.05
                    )
                    samples.extend(random.choices(other_symmetric_samples, k=augmentation_limit))
                    other_symmetry_count += augmentation_limit

            # Augment the data to balance the classes.
            if random_augmentation:
                augmentation_count = augmentation_amount(real_data_count)
                for _ in range(augmentation_count):
                    row = random.choice(samples)
                    samples.append((row[0], row[1], True))

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
        primary_key, required_transform, do_random_transform = self.primary_keys[idx]
        stroke_data = self.load_stroke_data(primary_key)
        # If a symmetric character was used, we will need to apply it's transformation.
        # We may have multiple options here.
        trans_mats = []
        if not required_transform.is_none():
            if required_transform.is_rotation():
                trans_mats.append(rotation_matrix(required_transform.angle))
            else:
                trans_mats.append(reflection_matrix(required_transform.angle))

        # Augment the data with a random transformation.
        # The transformation is applied to the strokes before converting them to an image.
        if do_random_transform:
            operation = random.randint(0, 2)
            if operation == 0:
                trans_mats.append(rotation_matrix(np.random.uniform(-5, 5)))
            elif operation == 1:
                trans_mats.append(
                    scale_matrix(np.random.uniform(0.9, 1), np.random.uniform(0.9, 1))
                )
            elif operation == 2:
                trans_mats.append(
                    skew_matrix(np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
                )

        # Apply the transformations to the stroke data.
        if trans_mats:
            stroke_data = apply_transformations(stroke_data, trans_mats)

        symbol_key = self.symbol_keys[idx]

        return stroke_data, symbol_key

    def __getitem__(self, idx):
        stroke_data, _ = self.load_transformed_strokes(idx)

        img = strokes_to_grayscale_image_cv2(stroke_data, self.image_size)

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


def strokes_to_grayscale_image_cv2(stroke_data: list[list[tuple[int, int]]], image_size: int):
    # Create a blank white image (grayscale)
    img = np.ones((image_size, image_size), dtype=np.uint8) * 255  # White background (255)

    # Scale down the strokes to fit the image size, adding an extra 5% padding.
    padding = 0.10
    scale = (image_size * (1 - padding)) / 1000
    offset = (image_size * padding) / 2
    stroke_data = [
        [(round(point[0] * scale + offset), round(point[1] * scale + offset)) for point in stroke]
        for stroke in stroke_data
    ]

    # Clear out duplicate consecutive points, so that they can be detected as single points.
    # Otherwise, a short segment could be scaled down to a single point, and then
    # not get the point handling treatment, resulting in nothing being drawn.
    for stroke in stroke_data:
        for i in range(len(stroke) - 1, 0, -1):
            if stroke[i] == stroke[i - 1]:
                stroke.pop(i)

    # Iterate through each stroke and draw it
    for stroke in stroke_data:
        if len(stroke) == 1:
            # Handle single point by adding a second point offset by 1
            point = stroke[0]
            stroke.append((point[0] + 1, point[1] + 1))

        for i in range(len(stroke) - 1):
            cv2.line(
                img, stroke[i], stroke[i + 1], color=(0,), thickness=1, lineType=cv2.LINE_AA
            )  # Draw black lines (0)
    return img


def rotation_matrix(angle: float, image_size: int = 1000) -> np.ndarray:
    """
    Rotate the stroke data by the given angle around the center of the image.
    Positive angles rotate counter-clockwise, negative angles rotate clockwise.
    The center is at coordinates (image_size / 2, image_size / 2), not specific to
    the image, since the image is expected to be centered in the frame.
    After rotation, the image may need to be scaled down to fit the original image size.

    :param angle: Angle in degrees to rotate the strokes.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Rotation matrix.
    """
    angle_rad = np.radians(-angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = np.array(
        [
            [cos_theta, -sin_theta, -x_offset * cos_theta + y_offset * sin_theta + x_offset],
            [sin_theta, cos_theta, -x_offset * sin_theta - y_offset * cos_theta + y_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def reflection_matrix(angle: float, image_size: int = 1000) -> np.ndarray:
    """
    Reflect the stroke data along an axis that passes through the center of the image
    at a specified angle. The center is at coordinates (image_size / 2, image_size / 2).
    Positive angles rotate the axis counter-clockwise, negative angles rotate clockwise.

    :param angle: Angle in degrees defining the axis to reflect across.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Reflection matrix.
    """
    angle_rad = np.radians(-angle)
    two_theta = 2 * angle_rad
    cos_2theta = np.cos(two_theta)
    sin_2theta = np.sin(two_theta)
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = np.array(
        [
            [cos_2theta, sin_2theta, x_offset - cos_2theta * x_offset - sin_2theta * y_offset],
            [sin_2theta, -cos_2theta, y_offset - sin_2theta * x_offset + cos_2theta * y_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def scale_matrix(scale_x: float, scale_y: float, image_size: int = 1000) -> np.ndarray:
    """
    Scale the stroke data along the x and y axes, relative to the center of the image.
    The center is at coordinates (image_size / 2, image_size / 2).

    :param scale_x: Scaling factor for the x-axis. Values less than 1 will shrink the x-axis.
    :param scale_y: Scaling factor for the y-axis. Values less than 1 will shrink the y-axis.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Scaling matrix.
    """
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = np.array(
        [
            [scale_x, 0, x_offset - scale_x * x_offset],
            [0, scale_y, y_offset - scale_y * y_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def skew_matrix(skew_x: float, skew_y: float, image_size: int = 1000) -> np.ndarray:
    """
    Skew the stroke data along the x and y axes, relative to the center of the image.
    The center is at coordinates (image_size / 2, image_size / 2).
    If the skewed strokes exceed the image boundaries, they are scaled down to fit.

    :param skew_x: Skew factor along the x-axis. Positive values shear to the right.
    :param skew_y: Skew factor along the y-axis. Positive values shear upwards.
    :param image_size: Size of the image the strokes are drawn on.
    :return: Skew matrix.
    """
    x_offset = image_size / 2
    y_offset = image_size / 2

    transformation = np.array(
        [
            [1, skew_x, -skew_x * y_offset],
            [skew_y, 1, -skew_y * x_offset],
            [0, 0, 1],
        ]
    )
    return transformation


def apply_transformations(
    stroke_data: list[list[tuple[int, int]]],
    transformations: list[np.ndarray] | np.ndarray,
    shuffle_transformations: bool = False,
    image_size: int = 1000,
) -> list[list[tuple[int, int]]]:
    """
    Apply a sequence of transformation matrices to the stroke data.
    If the transformed strokes exceed the image boundaries, they are scaled down to fit.
    Transformations are applied in the ascending order they are provided, unless shuffled.

    :param stroke_data: List of strokes, each stroke being a list of points.
    :param transformations: List of 3x3 transformation matrices to apply.
    :param shuffle_transformations: [Optional] If True, shuffle the order of transformations.
    :param image_size: [Optional] Size of the image the strokes are drawn on.
    :return: Transformed stroke data.
    """
    if not isinstance(transformations, list):
        transformations = [transformations]

    # Shuffle transformations if required
    if shuffle_transformations:
        generator = np.random.default_rng()
        generator.shuffle(transformations)

    # Combine all transformation matrices into a single matrix
    total_transformation = np.eye(3)
    for transformation in transformations:
        total_transformation = transformation @ total_transformation

    # Flatten all strokes into a single array and keep track of stroke lengths.
    all_points = []
    stroke_lengths = []
    for stroke in stroke_data:
        all_points.extend(stroke)
        stroke_lengths.append(len(stroke))

    points_array = np.array(all_points)
    ones = np.ones((points_array.shape[0], 1))
    homogeneous_points = np.hstack([points_array, ones])

    transformed_points = homogeneous_points @ total_transformation.T
    x_new, y_new = transformed_points[:, 0], transformed_points[:, 1]

    # Calculate the bounding box of the transformed points.
    # We may need to scale the image down to fit again.
    min_x, min_y = x_new.min(), y_new.min()
    max_x, max_y = x_new.max(), y_new.max()

    width, height = max_x - min_x, max_y - min_y
    if width == 0 or height == 0:
        scale = 1
    else:
        scale = min(image_size / width, image_size / height)

    if scale < 1:
        tx = (1 - scale) * image_size / 2
        ty = (1 - scale) * image_size / 2
        scaling_matrix = np.array(
            [
                [scale, 0, tx],
                [0, scale, ty],
                [0, 0, 1],
            ]
        )

        ones = np.ones((transformed_points.shape[0], 1))
        transformed_points = np.hstack([transformed_points[:, :2], ones])
        scaled_points = transformed_points @ scaling_matrix.T
        x_scaled = np.round(scaled_points[:, 0]).astype(int)
        y_scaled = np.round(scaled_points[:, 1]).astype(int)
    else:
        x_scaled = np.round(x_new).astype(int)
        y_scaled = np.round(y_new).astype(int)

    # Split the transformed points back into strokes.
    transformed_strokes = []
    idx = 0
    for length in stroke_lengths:
        stroke_points = list(zip(x_scaled[idx : idx + length], y_scaled[idx : idx + length]))
        transformed_strokes.append(stroke_points)
        idx += length

    return transformed_strokes


def tensorize_strokes(stroke_data: list[list[tuple[int, int]]], image_size: int):
    img = strokes_to_grayscale_image_cv2(stroke_data, image_size)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts the image to a PyTorch tensor (C, H, W) with values in [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to range [-1, 1]
        ]
    )
    img_tensor = transform(img)  # Output shape will be [1, H, W]

    # Expand the tensor to add a batch dimension.
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def dump_encoder(label_encoder: LabelEncoder, labels: list[str], path: Path | str):
    encoded_labels = label_encoder.transform(labels)
    # Create a decoder dictionary to map encoded labels back to symbols
    decoder: list[tuple[int, str]] = [
        (encoded, symbol) for encoded, symbol in zip(encoded_labels, labels)
    ]
    # Sort the decoder by encoded label for easy lookup, ascending.
    decoder.sort(key=lambda x: x[0])
    # Assert the encodings are consecutive integers starting from 0
    assert all(encoded == i for i, (encoded, _) in enumerate(decoder)), "Invalid label encodings"
    # Dump the sorted symbols to a plain text file.
    encoding_str = "\n".join((symbol for _, symbol in decoder))
    with open(path, "w") as file:
        file.write(encoding_str)


def load_decoder(path: Path) -> dict[int, str]:
    with open(path, "r") as file:
        decoder = {i: symbol.strip() for i, symbol in enumerate(file)}
    return decoder


def recalculate_encodings():
    """
    Encodings are the integer labels assigned to each symbol.
    These are required for the model to classify the symbols,
    since it only operates on integers.
    Each symbol is assigned a sequential integer encoding.
    These are dumped to a text file, where each line is a symbol.
    The line number-1 implies the encoding value (since it's 0-indexed).
    """
    symbols = ut.load_symbols()
    similar_symbols = ut.load_symbol_metadata_similarity()

    # Limit the number of classes to classify.
    symbol_keys = ut.select_leader_symbols(list(symbols.keys()), similar_symbols)

    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_keys)

    encoding_path = ut.get_encodings_path()
    dump_encoder(label_encoder, symbol_keys, encoding_path)


def recalculate_frequencies():
    symbols = ut.load_symbols()
    similar_symbols = ut.load_symbol_metadata_similarity()

    # database_path = "database/handtex.db"
    with ut.resource_path(training.database, "handtex.db") as path:
        database_path = path

    # Limit the number of classes to classify.
    leader_keys = ut.select_leader_symbols(list(symbols.keys()), similar_symbols)

    with resources.path(handtex.data.symbol_metadata, "symbol_frequency.csv") as path:
        frequencies_path = path

    with resources.path(handtex.data.symbol_metadata, "leader_symbol_frequency.csv") as path:
        leader_frequencies_path = path

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
    missing_symbols = set(symbols.keys()) - set(frequencies.keys())
    if missing_symbols:
        print(f"Missing frequencies for symbols:")
        for symbol in missing_symbols:
            print(symbol)
        frequencies.update({key: 0 for key in missing_symbols})

    with open(frequencies_path, "w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(frequencies.items())

    # Calculate new frequencies for the leader symbols
    leader_frequencies = {key: frequencies[key] for key in leader_keys}
    for leader in leader_keys:
        for similar in similar_symbols.get(leader, []):
            leader_frequencies[leader] += frequencies[similar]
    # Dump the new frequencies to a CSV file, sorted by frequency.
    sorted_frequencies = sorted(leader_frequencies.items(), key=lambda item: item[1], reverse=True)
    with open(leader_frequencies_path, "w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(sorted_frequencies)

    symbol_mean_freq = sum(frequencies.values()) / len(frequencies)
    leader_mean_freq = sum(leader_frequencies.values()) / len(leader_frequencies)
    print(f"Mean frequency of all symbols: {symbol_mean_freq}")
    print(f"Mean frequency of leader symbols: {leader_mean_freq}")


def main():
    # Test data loading.
    symbols = ut.load_symbols()
    similar_symbols = ut.load_symbol_metadata_similarity()
    symbol_keys = ut.select_leader_symbols(list(symbols.keys()), similar_symbols)
    self_symmetries = ut.load_symbol_metadata_self_symmetry()
    other_symmetries = ut.load_symbol_metadata_other_symmetry()

    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_keys)

    db_path = "database/handtex.db"
    image_size = 48

    # Create training and validation datasets and dataloaders
    all_samples = StrokeDataset(
        db_path,
        symbol_keys,
        similar_symbols,
        self_symmetries,
        other_symmetries,
        image_size,
        label_encoder,
        validation_split=0,
        train=True,
        shuffle=False,
    )

    # Show all the samples for a given symbol.
    symbol = "amssymb-OT1-_llcorner"
    assert symbol in symbol_keys, f"Symbol '{symbol}' not found in the dataset or not a leader"
    symbol_strokes = (
        all_samples.load_transformed_strokes(idx) for idx in all_samples.range_for_symbol(symbol)
    )
    for idx, (strokes, symbol) in enumerate(symbol_strokes):
        img = strokes_to_grayscale_image_cv2(strokes, image_size)
        cv2.imshow(f"{symbol} {idx}", img)
        # wait for a key press
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
