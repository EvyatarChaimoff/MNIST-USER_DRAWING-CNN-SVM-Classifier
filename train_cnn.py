import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from cnn_model import CNNModel

def train_cnn(train_imgs, train_lbls, dev_imgs, dev_lbls, batch_size=64, max_epochs=100, patience=5):
    print("Starting CNN model training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors and send to device
    train_imgs = torch.tensor(train_imgs, dtype=torch.float32).unsqueeze(1)
    train_lbls = torch.tensor(train_lbls, dtype=torch.long)
    dev_imgs = torch.tensor(dev_imgs, dtype=torch.float32).unsqueeze(1)
    dev_lbls = torch.tensor(dev_lbls, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_imgs, train_lbls), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(dev_imgs, dev_lbls), batch_size=batch_size)

    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(max_epochs), desc='Training CNN (Epochs)'):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in dev_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(probabilities, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_accuracy:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs("Models", exist_ok=True)
            torch.save(model.state_dict(), "Models/cnn_best_model.pth")
            print("Best model saved at Models/cnn_best_model.pth")
        else:
            epochs_no_improve += 1
            print(f"No validation loss improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Save plots
    os.makedirs("Plots/CNN", exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, 'r-', label='Train Loss')
    plt.plot(epochs_range, train_accuracies, 'b-', label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.savefig("Plots/CNN/Train Loss and Accuracy over Epochs.png")
    plt.close()

    plt.figure()
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.plot(epochs_range, val_accuracies, 'b-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Validation Loss and Accuracy')
    plt.legend()
    plt.savefig("Plots/CNN/Validation Loss and Accuracy over Epochs.png")
    plt.close()

    print("Training plots saved at Plots/CNN/")
    print("Training done!")
