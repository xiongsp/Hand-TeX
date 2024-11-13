import numpy as np
import cv2
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import sqlite3
import json
from sklearn.preprocessing import LabelEncoder


class StrokeDataset(Dataset):
    def __init__(
        self,
        db_path,
        symbol_keys: list[str],
        image_size: int,
        label_encoder: LabelEncoder,
        train: bool = True,
    ):
        """
        Args:
            db_path (str): Path to the SQLite database.
            symbol_keys (list): List of symbol keys to load stroke data for.
            image_size (tuple): Size of the generated images (width, height).
        """
        self.db_path = db_path
        self.image_size = image_size
        self.primary_keys = []
        self.symbol_keys = []
        self.train = train

        # Load primary keys and symbol keys from the database for the provided symbol keys
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for symbol_key in symbol_keys:
            cursor.execute("SELECT id FROM samples WHERE key = ?", (symbol_key,))
            rows = cursor.fetchall()
            split_idx = ceil(len(rows) * 0.1)
            if train:
                selected_rows = rows[split_idx:]  # Remaining 90% for training
            else:
                selected_rows = rows[:split_idx]  # First 10% for validation

            self.primary_keys.extend([row[0] for row in selected_rows])
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

    def __getitem__(self, idx):
        primary_key = self.primary_keys[idx]
        stroke_data = self.load_stroke_data(primary_key)
        img = strokes_to_grayscale_image_cv2(stroke_data, self.image_size)

        # Apply the transform to convert the image to a tensor and normalize it
        img_tensor = self.transform(img)
        label_tensor = torch.tensor(
            self.encoded_labels[idx], dtype=torch.long
        )  # Convert label to tensor
        return img_tensor, label_tensor  # Return image tensor and label

    def load_stroke_data(self, primary_key):
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


def tensorize_strokes(stroke_data: list[list[tuple[int, int]]], image_size: int):
    img = strokes_to_grayscale_image_cv2(stroke_data, image_size)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts the image to a PyTorch tensor (C, H, W) with values in [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to range [-1, 1]
        ]
    )
    img_tensor = transform(img)  # Output shape will be [1, H, W]

    # Expand the tensor to add a batch dimension, final shape should be [1, 1, 28, 28]
    img_tensor = img_tensor.unsqueeze(0)

    # Assert that the tensor has the correct shape for the model input
    assert img_tensor.shape == (
        1,
        1,
        28,
        28,
    ), f"Expected shape (1, 1, 28, 28), got {img_tensor.shape}"

    return img_tensor


def dump_encoder(label_encoder, labels, path="encodings.txt"):
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


def load_decoder(path="encodings.txt") -> dict[int, str]:
    with open(path, "r") as file:
        decoder = {i: symbol.strip() for i, symbol in enumerate(file)}
    return decoder


def main():
    # path = "../new_drawings/session1.json"
    # with open(path, "r") as file:
    #     data = json.load(file)

    # Example usage
    symbol_keys = [
        "latex2e-OT1-_xi",
        "latex2e-OT1-_sim",
        "latex2e-OT1-_infty",
        "latex2e-OT1-_approx",
        "dsfont-OT1-_mathds{R}",
        "latex2e-OT1-_partial",
        "latex2e-OT1-_equiv",
        "latex2e-OT1-_alpha",
        "latex2e-OT1-_sum",
        "latex2e-OT1-_int",
    ]
    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_keys)

    db_path = "dextify/detexify_rescaled.db"
    image_size = 28

    # Create training and validation datasets and dataloaders
    train_dataset = StrokeDataset(db_path, symbol_keys, image_size, label_encoder, train=True)
    validation_dataset = StrokeDataset(db_path, symbol_keys, image_size, label_encoder, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    # Iterate through the training dataloader
    for batch, labels in train_dataloader:
        print(batch.shape, labels[0], "...")  # Output the shape of the batch and labels
        # You can now use the batch for training a model

    # Iterate through the validation dataloader
    for batch, labels in validation_dataloader:
        print(batch.shape, labels[0], "...")  # Output the shape of the batch and labels
        # You can now use the batch for validation

    # Show one of the images (optional)
    stroke_data = train_dataset.load_stroke_data(train_dataset.primary_keys[0])
    img_cv2 = strokes_to_grayscale_image_cv2(stroke_data, image_size)
    cv2.imshow("Stroke Image", img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output_cv2.png", img_cv2)


if __name__ == "__main__":
    main()
