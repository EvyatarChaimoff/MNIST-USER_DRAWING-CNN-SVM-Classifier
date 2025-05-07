import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel, padding=1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel, padding=1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        """
        The transfer between convolutional layers and the final transfer from convolutional to fully connected:

        Each Conv2D layer applies the following formula to compute output size along one spatial dimension (height or width):
            Output = floor((Input + 2 * Padding - Kernel) / Stride) + 1

        Example for conv1:
            Input = 28, Padding = 1, Kernel = 3, Stride = 1
            Output = floor((28 + 2*1 - 3) / 1) + 1 = 28
            → Output shape: (32, 28, 28)

        After applying 2x2 MaxPooling:
            Output = floor(28 / 2) = 14
            → Shape becomes: (32, 14, 14)

        conv2 keeps the same padding and kernel:
            Input = 14, Padding = 1, Kernel = 3, Stride = 1
            Output = floor((14 + 2*1 - 3) / 1) + 1 = 14
            → Output shape: (64, 14, 14)

        After second 2x2 MaxPooling:
            Output = floor(14 / 2) = 7
            → Final conv output shape: (64, 7, 7)

        To connect to the first fully connected (Linear) layer, we flatten the output:
            64 * 7 * 7 = 3136 input features to fc1
        """

        #Fully connected layers
        self.fc1 = nn.Linear(7*7*64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for digits 0-9

    def forward(self, x):
        # Convolution -> ReLU -> MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

