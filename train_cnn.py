# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from cnn_model import CNNModel

# Train a CNN model with early stopping and save best-performing model
def train_cnn(train_imgs, train_lbls, dev_imgs, dev_lbls, batch_size=64, max_epochs=100, patience=5):
    print("Starting CNN model training...")

    # Select training device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert NumPy arrays to PyTorch tensors and add channel dimension
    train_imgs = torch.tensor(train_imgs, dtype=torch.float32).unsqueeze(1)  # shape: [N, 1, 28, 28]
    train_lbls = torch.tensor(train_lbls, dtype=torch.long)
    dev_imgs = torch.tensor(dev_imgs, dtype=torch.float32).unsqueeze(1)
    dev_lbls = torch.tensor(dev_lbls, dtype=torch.long)

    # Create data loaders for training and validation
    train_loader = DataLoader(TensorDataset(train_imgs, train_lbls), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(dev_imgs, dev_lbls), batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Variables to track early stopping and best model
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Lists to track metrics per epoch for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Main training loop
    for epoch in tqdm(range(max_epochs), desc='Training CNN (Epochs)'):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        # Loop over mini-batches
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()                   # Clear previous gradients
            outputs = model(images)                 # Forward pass
            loss = criterion(outputs, labels)       # Compute loss
            loss.backward()                         # Backpropagation
            optimizer.step()                        # Update weights

            running_loss += loss.item()

            # Collect predictions for accuracy calculation
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

        # Compute average training loss and accuracy for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for images, labels in dev_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)  # Accumulate total loss
                _,_
