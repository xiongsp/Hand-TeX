import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from training.image_gen import StrokeDataset, dump_encoder

import handtex.utils as ut


class CNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        """
        Define the layers of the convolutional neural network.

        :param num_classes: The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(CNN, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        # 1 input channel because images are grayscale.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)  # Apply fully connected layer
        return x


def check_accuracy(loader, model, device):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    if loader.dataset.train:
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

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = len(symbol_keys)
learning_rate = 0.001
batch_size = 64
num_epochs = 10  # Reduced for demonstration purposes

db_path = "dextify/detexify_rescaled.db"
image_size = 28

model_path = "cnn_model.pt"
encodings_path = "encodings.txt"


def main():

    label_encoder = LabelEncoder()
    label_encoder.fit(symbol_keys)

    dump_encoder(label_encoder, symbol_keys, encodings_path)

    # Create training and validation datasets and dataloaders
    train_dataset = StrokeDataset(db_path, symbol_keys, image_size, label_encoder, train=True)
    validation_dataset = StrokeDataset(db_path, symbol_keys, image_size, label_encoder, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    model = CNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
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

    # Final accuracy check on training and test sets
    check_accuracy(train_dataloader, model, device)
    check_accuracy(validation_dataloader, model, device)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
