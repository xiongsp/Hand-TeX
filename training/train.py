import matplotlib.pyplot as plt
import torch
from safetensors.torch import save_file
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import handtex.symbol_relations as sr
import handtex.utils as ut
from training.hyperparameters import (
    db_path,
    batch_size,
    num_epochs,
    image_size,
    learning_rate,
)
from training.data_loader import (
    StrokeDataset,
    recalculate_frequencies,
    build_stroke_cache,
)
from training.model import CNN


def save_encoder(label_encoder: LabelEncoder, leader_keys: list[str]):
    """
    Encodings are the integer labels assigned to each symbol.
    These are required for the model to classify the symbols,
    since it only operates on integers.
    Each symbol is assigned a sequential integer encoding.
    These are dumped to a text file, where each line is a symbol.
    The line number-1 implies the encoding value (since it's 0-indexed).
    """
    encoded_labels = label_encoder.transform(leader_keys)
    # Create a decoder dictionary to map encoded labels back to symbols
    decoder: list[tuple[int, str]] = [
        (encoded, symbol) for encoded, symbol in zip(encoded_labels, leader_keys)
    ]
    # Sort the decoder by encoded label for easy lookup, ascending.
    decoder.sort(key=lambda x: x[0])

    # Assert the encodings are consecutive integers starting from 0
    assert all(encoded == i for i, (encoded, _) in enumerate(decoder)), "Invalid label encodings"

    # Dump the sorted symbols to a plain text file.
    encoding_path = ut.get_encodings_path()
    encoding_str = "\n".join((symbol for _, symbol in decoder))
    with open(encoding_path, "w") as file:
        file.write(encoding_str)


def main(resume_from_checkpoint=False):
    symbol_data = sr.SymbolData()

    num_classes = len(symbol_data.leaders)

    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_data.leaders)

    recalculate_frequencies()

    stroke_cache = build_stroke_cache(db_path)

    random_seed = 0

    # Create training and validation datasets and dataloaders
    train_dataset = StrokeDataset(
        db_path,
        symbol_data,
        image_size,
        label_encoder,
        random_seed,
        validation_split=0.2,
        sample_limit=500,
        train=True,
        stroke_cache=stroke_cache,
    )
    validation_dataset = StrokeDataset(
        db_path,
        symbol_data,
        image_size,
        label_encoder,
        random_seed,
        validation_split=0.2,
        sample_limit=100,
        train=False,
        stroke_cache=stroke_cache,
    )

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=True, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(num_classes=num_classes, image_size=image_size).to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Lists to store training and validation metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0.0
    checkpoint_path = "best_model_checkpoint.chkpt"

    start_epoch = 0

    # Optionally load from checkpoint
    if resume_from_checkpoint:
        print(f"Attempting to resume from checkpoint at {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_accuracy = checkpoint["best_val_accuracy"]
        start_epoch = checkpoint["epoch"]

        # Load train and validation metrics if they exist in the checkpoint
        if "train_losses" in checkpoint:
            train_losses = checkpoint["train_losses"]
            train_accuracies = checkpoint["train_accuracies"]
            val_losses = checkpoint["val_losses"]
            val_accuracies = checkpoint["val_accuracies"]
        else:
            train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

        print(
            f"Resumed from epoch {start_epoch} with best validation accuracy: {best_val_accuracy:.2f}%"
        )

    all_preds, all_targets = None, None

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        # Set model to training mode
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (data, targets) in enumerate(tqdm(train_dataloader)):
            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * data.size(0)

            # Calculate accuracy
            _, predicted = torch.max(scores.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets in validation_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                val_loss = criterion(outputs, targets)
                val_running_loss += val_loss.item() * data.size(0)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_epoch_loss = val_running_loss / len(validation_dataloader)
        val_epoch_accuracy = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        print(
            f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%"
        )

        # Save checkpoint if validation accuracy improves
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "train_losses": train_losses,
                    "train_accuracies": train_accuracies,
                    "val_losses": val_losses,
                    "val_accuracies": val_accuracies,
                },
                checkpoint_path,
            )

            print(f"Checkpoint saved at {checkpoint_path}")

        # Step the scheduler
        scheduler.step()

    # Save the model.
    model_path = ut.get_model_path()
    save_file(model.state_dict(), model_path)
    save_encoder(label_encoder, symbol_data.leaders)
    print(f"Model saved to {model_path}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

    # We have no stats.
    if all_preds is None:
        return

    from collections import Counter

    misclassifications = [
        (target, pred) for target, pred in zip(all_targets, all_preds) if target != pred
    ]

    # Counter for misclassified pairs
    misclass_counter = Counter(misclassifications)

    # Dictionary to hold the total count for each symbol in the dataset (use the symbol directly as the key)
    total_counts = {
        symbol: len(validation_dataset.range_for_symbol(symbol))
        for symbol in label_encoder.classes_
    }

    # List to hold misclassifications with their percentage
    misclass_percentage = []

    # Calculate the misclassification percentage
    for (actual, predicted), count in misclass_counter.items():
        actual_label = label_encoder.inverse_transform([actual])[
            0
        ]  # Convert 'actual' index to label
        total_count = total_counts.get(
            actual_label, 0
        )  # Use the label directly to lookup in total_counts
        if total_count > 0:
            percentage = (count / total_count) * 100  # Calculate percentage misclassification
            misclass_percentage.append(((actual_label, predicted), percentage, count))

    # Sort by misclassification percentage
    top_misclassified_by_percentage = sorted(misclass_percentage, key=lambda x: x[1], reverse=True)[
        :50
    ]

    # Print top 50 most misclassified symbols by percentage
    print("\nTop 50 Most Misclassified Symbols (by Percentage):")
    for (actual, predicted), percentage, count in top_misclassified_by_percentage:
        predicted_label = label_encoder.inverse_transform([predicted])[
            0
        ]  # Convert 'predicted' index to label
        print(
            f"Actual: {actual}, Predicted: {predicted_label}, Count: {count}, Percentage: {percentage:.2f}%"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a CNN model with optional checkpointing.")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from the latest checkpoint."
    )
    args = parser.parse_args()

    main(resume_from_checkpoint=args.resume)
