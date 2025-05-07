import torch
from cnn_model import CNNModel
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import joblib
import numpy as np
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from utilities import save_plot_confusion_matrix, save_metrics

def test_cnn(test_imgs, test_lbls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    model.load_state_dict(torch.load("Models/cnn_best_model.pth", map_location=device))
    model.eval()

    test_imgs = torch.tensor(test_imgs, dtype=torch.float32).unsqueeze(1)
    test_lbls = torch.tensor(test_lbls, dtype=torch.long)

    start_time = time.time()

    with tqdm(total=1, desc="Testing CNN") as pbar:
        outputs = model(test_imgs.to(device)).cpu()
        probabilities = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probabilities, 1)
        pbar.update(1)

    elapsed_time = time.time() - start_time 
    print(f"\nTest time: {elapsed_time:.2f} seconds")

    acc = accuracy_score(test_lbls, preds)
    cm = confusion_matrix(test_lbls, preds)
    precision = precision_score(test_lbls, preds, average='weighted', zero_division=0)
    recall = recall_score(test_lbls, preds, average='weighted', zero_division=0)

    print("CNN Test Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print("Confusion Matrix:\n", cm)

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "Confusion Matrix": cm,
        "Test Time (s)": elapsed_time
    }
    save_metrics(metrics, "CNN")
    save_plot_confusion_matrix(cm, "CNN")
    # Find indices of misclassified examples
    misclassified_indices = (preds != test_lbls).nonzero(as_tuple=False).squeeze().tolist()

    # Ensure the list is always a list (even if one element)
    if isinstance(misclassified_indices, int):
        misclassified_indices = [misclassified_indices]

    # Loop over the first 3 misclassified examples (or fewer if there are less than 3)
    for i in range(min(3, len(misclassified_indices))):
        misclassified_idx = misclassified_indices[i]

        # Get the image, true label, and predicted label
        misclassified_img = test_imgs[misclassified_idx].squeeze().numpy()
        true_label = test_lbls[misclassified_idx].item()
        pred_label = preds[misclassified_idx].item()

        # Create directory
        os.makedirs("False predict/CNN", exist_ok=True)

        # Save the image
        filename = f"False predict/CNN/true_{true_label}_pred_{pred_label}_{i+1}.png"
        plt.imshow(misclassified_img, cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
        plt.savefig(filename)
        plt.close()

        # Print confirmation
        print(f"Saved misclassified example: {filename}")


def test_svm(test_imgs, test_lbls):
    kernels = ['linear', 'rbf', 'poly']

    np.save("data_numpy/test_images_SVM.npy", test_imgs)
    np.save("data_numpy/test_labels_SVM.npy", test_imgs)

    test_imgs = test_imgs.reshape(test_imgs.shape[0], -1)

    for kernel in kernels:
        print(f"\nTesting SVM model ({kernel})...")
        model = joblib.load(f"Models/svm_{kernel}.pkl")

        start_time = time.time()

        with tqdm(total=1, desc=f"Testing SVM ({kernel})") as pbar:
            preds = model.predict(test_imgs)
            pbar.update(1)

        elapsed_time = time.time() - start_time
        print(f"Test time for {kernel}: {elapsed_time:.2f} seconds")

        acc = accuracy_score(test_lbls, preds)
        cm = confusion_matrix(test_lbls, preds)
        precision = precision_score(test_lbls, preds, average='weighted', zero_division=0)
        recall = recall_score(test_lbls, preds, average='weighted', zero_division=0)

        print(f"SVM ({kernel}) Test Metrics:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print("Confusion Matrix:\n", cm)

        metrics = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": cm,
            "Test Time (s)": elapsed_time
        }
        save_metrics(metrics, f"SVM_{kernel}")
        save_plot_confusion_matrix(cm, f"SVM/{kernel}")
        # Find 3 misclassified example
    # Find indices of misclassified examples
        misclassified_indices = np.where(preds != test_lbls)[0]

        # Loop over the first 3 misclassified examples (or fewer if there are less than 3)
        for i in range(min(3, len(misclassified_indices))):
            misclassified_idx = misclassified_indices[i]

            # Get the image, true label, and predicted label
            misclassified_img = test_imgs[misclassified_idx].reshape(28, 28)
            true_label = test_lbls[misclassified_idx]
            pred_label = preds[misclassified_idx]

            # Create directory
            os.makedirs(f"False predict/SVM/{kernel}", exist_ok=True)

            # Save the image
            filename = f"False predict/SVM/{kernel}/true_{true_label}_pred_{pred_label}_{i+1}.png"
            plt.imshow(misclassified_img, cmap='gray')
            plt.title(f"True: {true_label}, Pred: {pred_label}")
            plt.axis('off')
            plt.savefig(filename)
            plt.close()

            # Print confirmation
            print(f"Saved misclassified example: {filename}")




