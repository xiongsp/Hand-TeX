import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import handtex.utils as ut
from training.image_gen import (
    StrokeDataset,
    recalculate_frequencies,
    recalculate_encodings,
)


class CNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        """
        Define the layers of the convolutional neural network.

        :param num_classes: The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(CNN, self).__init__()

        # Image size: 48 -> 24 -> 12 after each pooling.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the neural network.

        :param x: The input tensor.
        :return: The output tensor after passing through the network.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(x))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def check_accuracy(loader: DataLoader, model: nn.Module, device: str):
    """
    Checks the accuracy of the model on the given dataset loader.

    :param loader: The DataLoader for the dataset to check accuracy on.
    :param model: The neural network model.
    :param device: The device to run the model on (CPU or GPU).
    """
    if loader.dataset.train:  # noqa
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    model.train()  # Set the model back to training mode


symbols = ut.load_symbols()
symbol_keys = list(symbols.keys())

similar_symbols = ut.load_symbol_metadata_similarity()
self_symmetries = ut.load_symbol_metadata_self_symmetry()
other_symmetries = ut.load_symbol_metadata_other_symmetry()

# Limit the number of classes to classify.
symbol_keys = ut.select_leader_symbols(symbol_keys, similar_symbols)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

num_classes = len(symbol_keys)
learning_rate = 0.001
batch_size = 64
num_epochs = 17

db_path = "database/handtex.db"
image_size = 48


def main():

    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_keys)

    recalculate_frequencies()
    recalculate_encodings()

    # Create training and validation datasets and dataloaders
    train_dataset = StrokeDataset(
        db_path,
        symbol_keys,
        similar_symbols,
        self_symmetries,
        other_symmetries,
        image_size,
        label_encoder,
        validation_split=0.2,
        train=True,
    )
    validation_dataset = StrokeDataset(
        db_path,
        symbol_keys,
        similar_symbols,
        self_symmetries,
        other_symmetries,
        image_size,
        label_encoder,
        validation_split=0.2,
        train=False,
    )

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=True)

    model = CNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Lists to store training and validation metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
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

        val_epoch_loss = val_running_loss / len(validation_dataloader)
        val_epoch_accuracy = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        print(
            f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%"
        )

        # Step the scheduler
        scheduler.step()

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

    # torch.save(model.state_dict(), model_path)
    model_path = ut.get_model_path()
    save_file(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Final accuracy check on training and test sets
    check_accuracy(train_dataloader, model, device)
    check_accuracy(validation_dataloader, model, device)


if __name__ == "__main__":
    main()
