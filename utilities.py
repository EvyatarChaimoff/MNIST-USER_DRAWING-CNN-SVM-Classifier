# Import required libraries
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Group images by their labels using a dictionary of lists
def group_by_label(images, labels):
    groups = defaultdict(list)
    for img, label in zip(images, labels):
        groups[label].append(img)
    # Convert each list to a NumPy array for easier handling
    for label in groups:
        groups[label] = np.array(groups[label])
    return groups

# Plot label distribution as a bar chart for a given dataset
def plot_distribution(groups, set_name):
    labels = sorted(groups.keys())
    counts = [groups[label].shape[0] for label in labels]
    plt.figure()  # Create a new figure for each set
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Number of Samples')
    plt.title(f'{set_name} Set Label Distribution')
    plt.xticks(labels)

# Create a single composite image with one example per digit (0-9)
def create_sample_composite(groups):
    sample_images = [groups[label][0] for label in sorted(groups.keys())]  # Take the first sample from each label
    composite = np.concatenate(sample_images, axis=1)  # Join images horizontally
    return composite

# Load, split, normalize, and save MNIST dataset
def prepration():
    # Load raw MNIST dataset from IDX files
    train_images = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
    test_images = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')

    print("Loaded training set:", train_images.shape, train_labels.shape)
    print("Loaded test set:", test_images.shape, test_labels.shape)

    # Shuffle and split training data into training and development sets (90/10 split)
    num_train = train_images.shape[0]
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    split = int(0.9 * num_train)
    train_idx = indices[:split]
    dev_idx = indices[split:]

    train_images_split = train_images[train_idx]
    train_labels_split = train_labels[train_idx]
    dev_images = train_images[dev_idx]
    dev_labels = train_labels[dev_idx]

    print("After split:")
    print(" - Train set:", train_images_split.shape, train_labels_split.shape)
    print(" - Dev set:", dev_images.shape, dev_labels.shape)

    # Flatten the images for normalization (from 28x28 to 784)
    train_flat = train_images_split.reshape(train_images_split.shape[0], -1)
    dev_flat = dev_images.reshape(dev_images.shape[0], -1)
    test_flat = test_images.reshape(test_images.shape[0], -1)

    # Normalize the pixel values using z-score normalization
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat).reshape(train_images_split.shape)
    dev_scaled = scaler.transform(dev_flat).reshape(dev_images.shape)
    test_scaled = scaler.transform(test_flat).reshape(test_images.shape)

    # Save the scaler object for future normalization (e.g., user input)
    os.makedirs("Scalers", exist_ok=True)
    joblib.dump(scaler, "Scalers/mnist_scaler.pkl")

    # Save preprocessed datasets as NumPy arrays
    os.makedirs("data_numpy", exist_ok=True)
    np.save("data_numpy/train_images.npy", train_scaled)
    np.save("data_numpy/train_labels.npy", train_labels_split)
    np.save("data_numpy/dev_images.npy", dev_scaled)
    np.save("data_numpy/dev_labels.npy", dev_labels)
    np.save("data_numpy/test_images.npy", test_scaled)
    np.save("data_numpy/test_labels.npy", test_labels)

    print("Saved all normalized datasets to 'data_numpy/'")

    # Return datasets for further use in training/testing
    return train_scaled, train_labels_split, dev_scaled, dev_labels, test_scaled, test_labels

# Load preprocessed (saved) datasets from disk
def load_prepared_data():
    train_imgs = np.load('data_numpy/train_images.npy')
    train_lbls = np.load('data_numpy/train_labels.npy')
    dev_imgs = np.load('data_numpy/dev_images.npy')
    dev_lbls = np.load('data_numpy/dev_labels.npy')
    test_imgs = np.load('data_numpy/test_images.npy')
    test_lbls = np.load('data_numpy/test_labels.npy')
    return train_imgs, train_lbls, dev_imgs, dev_lbls, test_imgs, test_lbls

# Visualize dataset distribution and example digits
def visualize_data(train_imgs, train_lbls, dev_imgs, dev_lbls, test_imgs, test_lbls, isSVM=False):
    # Group images by label
    train_groups = group_by_label(train_imgs, train_lbls)
    dev_groups = group_by_label(dev_imgs, dev_lbls)
    test_groups = group_by_label(test_imgs, test_lbls)

    # Plot label distribution
    plot_distribution(train_groups, "Training")
    if not isSVM:  # Skip dev distribution if only training SVM (SVM doesn't use dev set)
        plot_distribution(dev_groups, "Development")
    plot_distribution(test_groups, "Test")

    # Display sample image from each digit (0-9)
    sample_composite = create_sample_composite(train_groups)
    plt.figure()
    plt.imshow(sample_composite, cmap='gray')
    plt.title("Training Set Sample Images (Concatenated)")
    plt.axis('off')
    plt.show()

# Save confusion matrix plot to disk
def save_plot_confusion_matrix(cm, model_name):
    os.makedirs(f"Plots/{model_name}", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Annotate each cell with its value
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"Plots/{model_name}/confusion_matrix.png")
    plt.close()

# Save evaluation metrics (accuracy, precision, recall, etc.) to file
def save_metrics(metrics, model_name):
    os.makedirs(f"Metrics/{model_name}", exist_ok=True)
    with open(f"Metrics/{model_name}/metrics.txt", "w") as f:
        for key, value in metrics.items():
            if key == "Confusion Matrix":
                f.write(f"{key}:\n{value}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")

# Print best SVM model parameters from saved files
def display_best_svm_models():
    print("Loading best SVM models...")
    
    # Dictionary mapping kernel names to model file paths
    model_paths = {
        'Linear': 'Models/svm_linear.pkl',
        'RBF': 'Models/svm_rbf.pkl',
        'Polynomial': 'Models/svm_poly.pkl'
    }

    print("The best SVM models are:")
    for kernel_name, model_path in model_paths.items():
        model = joblib.load(model_path)
        params = model.get_params()

        # Print relevant parameters for each kernel
        if kernel_name == 'Linear':
            print(f"{kernel_name} - C = {params['C']}")
        elif kernel_name == 'RBF':
            print(f"{kernel_name} - C = {params['C']}, Gamma = {model._gamma}")
        elif kernel_name == 'Polynomial':
            print(f"{kernel_name} - C = {params['C']}, Gamma = {model._gamma}, Degree = {params['degree']}")
