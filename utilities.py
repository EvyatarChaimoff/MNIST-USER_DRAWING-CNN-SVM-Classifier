import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from sklearn.preprocessing import StandardScaler
import joblib
# 3. Group (sort) images by their labels for each set
def group_by_label(images, labels):
    groups = defaultdict(list)
    for img, label in zip(images, labels):
        groups[label].append(img)
    # Convert lists to numpy arrays for convenience
    for label in groups:
        groups[label] = np.array(groups[label])
    return groups
# 4. Function to plot label distribution in a new figure
def plot_distribution(groups, set_name):
    labels = sorted(groups.keys())
    counts = [groups[label].shape[0] for label in labels]
    plt.figure()  # Create a new figure
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Number of Samples')
    plt.title(f'{set_name} Set Label Distribution')
    plt.xticks(labels)

# 5. Function to create a composite image from one sample per label
def create_sample_composite(groups):
    # Pick the first sample image from each label (assumes labels 0-9)
    sample_images = [groups[label][0] for label in sorted(groups.keys())]
    composite = np.concatenate(sample_images, axis=1)  # Concatenate horizontally
    return composite



def prepration():
    # 1. Load MNIST data from IDX files
    train_images = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')

    test_images = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')

    print("Loaded training set:", train_images.shape, train_labels.shape)
    print("Loaded test set:", test_images.shape, test_labels.shape)

    # 2. Split the training set into training and development sets
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

    # 3. Flatten images for normalization
    train_flat = train_images_split.reshape(train_images_split.shape[0], -1)
    dev_flat = dev_images.reshape(dev_images.shape[0], -1)
    test_flat = test_images.reshape(test_images.shape[0], -1)

    # 4. Normalize using StandardScaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat).reshape(train_images_split.shape)
    dev_scaled = scaler.transform(dev_flat).reshape(dev_images.shape)
    test_scaled = scaler.transform(test_flat).reshape(test_images.shape)

    # 5. Save scaler for later use
    os.makedirs("Scalers", exist_ok=True)
    import joblib
    joblib.dump(scaler, "Scalers/mnist_scaler.pkl")

    # 6. Save normalized datasets
    os.makedirs("data_numpy", exist_ok=True)

    np.save("data_numpy/train_images.npy", train_scaled)
    np.save("data_numpy/train_labels.npy", train_labels_split)
    np.save("data_numpy/dev_images.npy", dev_scaled)
    np.save("data_numpy/dev_labels.npy", dev_labels)
    np.save("data_numpy/test_images.npy", test_scaled)
    np.save("data_numpy/test_labels.npy", test_labels)

    print("Saved all normalized datasets to 'data_numpy/'")

    # 7. Return parameters
    return train_scaled, train_labels_split, dev_scaled, dev_labels, test_scaled, test_labels


def load_prepared_data():
    train_imgs = np.load('data_numpy/train_images.npy')
    train_lbls = np.load('data_numpy/train_labels.npy')
    dev_imgs = np.load('data_numpy/dev_images.npy')
    dev_lbls = np.load('data_numpy/dev_labels.npy')
    test_imgs = np.load('data_numpy/test_images.npy')
    test_lbls = np.load('data_numpy/test_labels.npy')
    
    return train_imgs, train_lbls, dev_imgs, dev_lbls, test_imgs, test_lbls

def visualize_data(train_imgs, train_lbls, dev_imgs, dev_lbls, test_imgs, test_lbls, isSVM = False):
    train_groups = group_by_label(train_imgs, train_lbls)
    dev_groups = group_by_label(dev_imgs, dev_lbls)
    test_groups = group_by_label(test_imgs, test_lbls)

    plot_distribution(train_groups, "Training")
    if not isSVM:
        plot_distribution(dev_groups, "Development")
    plot_distribution(test_groups, "Test")

    sample_composite = create_sample_composite(train_groups)
    plt.figure()
    plt.imshow(sample_composite, cmap='gray')
    plt.title("Training Set Sample Images (Concatenated)")
    plt.axis('off')

    plt.show()
def save_plot_confusion_matrix(cm, model_name):
    os.makedirs(f"Plots/{model_name}", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Add numbers in cells
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"Plots/{model_name}/confusion_matrix.png")
    plt.close()

def save_metrics(metrics, model_name):
    os.makedirs(f"Metrics/{model_name}", exist_ok=True)
    with open(f"Metrics/{model_name}/metrics.txt", "w") as f:
        for key, value in metrics.items():
            if key == "Confusion Matrix":
                f.write(f"{key}:\n{value}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")

def display_best_svm_models():
    print("Loading best SVM models...")
    # Load the SVM models
    # Note: Ensure that the paths are correct and the models are saved in the specified directory.
    #saved models
    model_paths = {
        'Linear': 'Models/svm_linear.pkl',
        'RBF': 'Models/svm_rbf.pkl',
        'Polynomial': 'Models/svm_poly.pkl'
    }
    print("The best SVM models are:")
    for kernel_name, model_path in model_paths.items():
        model = joblib.load(model_path)
        params = model.get_params()
        
        if kernel_name == 'Linear':
            print(f"{kernel_name} - C = {params['C']}")
        elif kernel_name == 'RBF':
            print(f"{kernel_name} - C = {params['C']}, Gamma = {model._gamma}")
        elif kernel_name == 'Polynomial':
            print(f"{kernel_name} - C = {params['C']}, Gamma = {model._gamma}, Degree = {params['degree']}")