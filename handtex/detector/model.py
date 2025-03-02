import torch
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int) -> None:
        """
        Define the layers of the convolutional neural network.

        :param num_classes: The number of classes we want to predict, in our case 10 (digits 0 to 9).
        :param image_size: The size of the input image.
        """
        super(CNN, self).__init__()

        self.image_size = image_size

        # Tuned by Optuna
        layer1 = 31
        layer2 = 52
        layer3 = 92
        layer4 = 122
        layer5 = 178

        self.last_conv = layer5

        # Image size is cut in half with each pooling.
        # This makes the fully connected layer have x * image_size/8 * image_size/8 input features.
        self.conv1 = nn.Conv2d(1, layer1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(layer1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(layer1, layer2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(layer2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(layer2, layer3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(layer3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(layer3, layer4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(layer4)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(layer4, layer5, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(layer5)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3301385640124735)

        self.fc1 = nn.Linear(self.last_conv, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the neural network.

        :param x: The input tensor.
        :return: The output tensor after passing through the network.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)

        x = self.dropout(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, self.last_conv)
        x = self.fc1(x)
        return x
