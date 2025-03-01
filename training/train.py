from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import torch
from safetensors.torch import save_file
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

import handtex.symbol_relations as sr
import handtex.utils as ut
from training.hyperparameters import (
    db_path,
    batch_size,
    num_epochs,
    learning_rate,
    weight_decay,
    step_size,
    gamma,
)
from handtex.detector.image_gen import IMAGE_SIZE
from training.data_loader import (
    StrokeDataset,
    DataSplit,
    recalculate_frequencies,
    build_stroke_cache,
)
from handtex.detector.model import CNN


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

    split_percentages = {
        DataSplit.TRAIN: 50,
        DataSplit.VALIDATION: 20,
        DataSplit.TEST: 30,
    }
    assert sum(split_percentages.values()) == 100, "Split points must sum to 100"

    # Create training, validation, and test datasets.
    train_dataset = StrokeDataset(
        db_path,
        symbol_data,
        IMAGE_SIZE,
        label_encoder,
        random_seed,
        split=DataSplit.TRAIN,
        split_percentages=split_percentages,
        class_limit=200,
        stroke_cache=stroke_cache,
    )
    validation_dataset = StrokeDataset(
        db_path,
        symbol_data,
        IMAGE_SIZE,
        label_encoder,
        random_seed,
        split=DataSplit.VALIDATION,
        split_percentages=split_percentages,
        class_limit=80,
        stroke_cache=stroke_cache,
    )
    test_dataset = StrokeDataset(
        db_path,
        symbol_data,
        IMAGE_SIZE,
        label_encoder,
        random_seed,
        split=DataSplit.TEST,
        split_percentages=split_percentages,
        class_limit=120,
        stroke_cache=stroke_cache,
    )

    # Create dataloaders.
    # For training we shuffle, but for validation/test we use shuffle=False.
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(num_classes=num_classes, image_size=IMAGE_SIZE).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Lists to store per-epoch metrics.
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    test_losses, test_accuracies, test_f1s = [], [], []

    best_val_accuracy = 0.0
    checkpoint_path = "best_model_checkpoint.chkpt"
    start_epoch = 0

    if resume_from_checkpoint:
        print(f"Attempting to resume from checkpoint at {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_accuracy = checkpoint["best_val_accuracy"]
        start_epoch = checkpoint["epoch"]

        # Load stored metrics if available.
        train_losses = checkpoint.get("train_losses", [])
        train_accuracies = checkpoint.get("train_accuracies", [])
        val_losses = checkpoint.get("val_losses", [])
        val_accuracies = checkpoint.get("val_accuracies", [])

        print(
            f"Resumed from epoch {start_epoch} with best validation accuracy: {best_val_accuracy:.2f}%"
        )

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, targets in tqdm(train_dataloader, desc="Training"):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(scores.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Note: using dataset length gives you the total number of samples.
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

        # Validation phase.
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data, targets in validation_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_epoch_loss = val_running_loss / len(validation_dataloader)
        val_epoch_accuracy = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        print(
            f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%"
        )

        # Test phase.
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for data, targets in test_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                test_running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
                test_preds.extend(predicted.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

        test_epoch_loss = test_running_loss / len(test_dataloader)
        test_epoch_accuracy = 100 * test_correct / test_total
        test_losses.append(test_epoch_loss)
        test_accuracies.append(test_epoch_accuracy)
        # Compute macro F1 score.
        epoch_f1 = f1_score(test_targets, test_preds, average="macro")
        test_f1s.append(epoch_f1)
        print(
            f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.2f}%, Test F1 (macro): {epoch_f1:.4f}"
        )

        # Save checkpoint if validation accuracy improves.
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

        scheduler.step()

    # Save the final model.
    model_path = ut.get_model_path()
    save_file(model.state_dict(), model_path)
    save_encoder(label_encoder, symbol_data.leaders)
    print(f"Model saved to {model_path}")

    # Plot loss over epochs.
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot accuracy over epochs (including test accuracy).
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()

    # Final evaluation on the test set to compute per-symbol F1 scores.
    model.eval()
    final_test_preds = []
    final_test_targets = []
    with torch.no_grad():
        for data, targets in test_dataloader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            final_test_preds.extend(predicted.cpu().numpy())
            final_test_targets.extend(targets.cpu().numpy())

    # Compute per-class F1 scores.
    f1_per_class = f1_score(final_test_targets, final_test_preds, average=None)
    # Count the number of test samples per class.
    counts = Counter(final_test_targets)
    # Get the symbol labels corresponding to each class index.
    symbols = label_encoder.inverse_transform(list(range(num_classes)))
    # Create a list of tuples: (symbol, count, F1 score)
    symbol_f1 = [(symbols[i], counts.get(i, 0), f1_per_class[i]) for i in range(num_classes)]
    # Sort the symbols in descending order by count.
    symbol_f1_sorted = sorted(symbol_f1, key=lambda x: x[1], reverse=True)

    # Plot the per-symbol F1 scores.
    plt.figure(figsize=(12, 6))
    symbol_names = [x[0] for x in symbol_f1_sorted]
    symbol_f1_scores = [x[2] for x in symbol_f1_sorted]
    plt.bar(symbol_names, symbol_f1_scores)
    plt.xlabel("Symbol")
    plt.ylabel("F1 Score")
    plt.title("Per-Symbol F1 Scores (sorted by count)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Count the total occurrences of each class (as indices) in the test set.
    total_counts = Counter(final_test_targets)

    # Build a counter for misclassified pairs (each pair is (actual, predicted)).
    misclassifications = [
        (target, pred)
        for target, pred in zip(final_test_targets, final_test_preds)
        if target != pred
    ]
    misclass_counter = Counter(misclassifications)

    # List to hold misclassification data: ((actual, predicted), percentage, misclassified_count)
    misclass_percentage = []
    for (actual, predicted), mis_count in misclass_counter.items():
        # Total count of the actual class in the test set.
        total = total_counts.get(actual, 0)
        if total > 0:
            percentage = (mis_count / total) * 100
            misclass_percentage.append(((actual, predicted), percentage, mis_count))

    # Sort by misclassification percentage (highest first) and take the top 50.
    top_misclassified_by_percentage = sorted(misclass_percentage, key=lambda x: x[1], reverse=True)[
        :50
    ]

    print("\nTop 50 Most Misclassified Symbols (by Percentage):")
    for (actual, predicted), percentage, count in top_misclassified_by_percentage:
        # Convert indices back to symbol labels.
        actual_label = label_encoder.inverse_transform([actual])[0]
        predicted_label = label_encoder.inverse_transform([predicted])[0]
        print(
            f"Actual: {actual_label}, Predicted: {predicted_label}, Count: {count}, Percentage: {percentage:.2f}%"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a CNN model with optional checkpointing.")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from the latest checkpoint."
    )
    args = parser.parse_args()

    main(resume_from_checkpoint=args.resume)
