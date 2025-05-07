# MNIST and User Digit Drawing Classification using CNN and SVM


## Setting Up the Project
To run the project successfully, follow the steps below.  
All terminal commands are written for **Git Bash** (installed with Git on Windows), but will also work in standard command lines (CMD, PowerShell, or VS Code terminal) if Python and Git is properly configured.

---

### Step 1: Clone the Repository

Open Git Bash (or your terminal of choice) and run:

```bash
git clone https://github.com/EvyatarChaimoff/MNIST-USER_DRAWING-CNN-SVM-Classifier.git
```
Then:
```
cd MNIST-USER_DRAWING-CNN-SVM-Classifier
```

---

### Step 2: Extract Required Files (Not through the terminal)

The repository includes compressed files (`.rar`) that must be extracted before running the project.

#### 2.1 Extract the Trained Models
Locate `Models.rar` in the following path:
```
MNIST-USER_DRAWING-CNN-SVM-Classifier/Models/Models.rar
```
Use **WinRAR** or **7-Zip** to right-click and choose:
> **Extract Here**  
This will generate model files like `cnn_best_model.pth` in the `Models/` directory.

#### 2.2 Extract the Data
Locate `data.rar` in:
```
MNIST-USER_DRAWING-CNN-SVM-Classifier/data/data.rar
```
Again, right-click → **Extract Here**  
This will extract `.npy` files such as `X_train.npy`, `y_train.npy` into the `data/` directory.

> Make sure the `Scalers/` folder (used for normalization) also exists. If missing, it may be included within the `Models.rar` or provided separately.

---

### Step 3: Install Python Dependencies (In terminal)

Run the following command inside the project directory:

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries for running the project (e.g., PyTorch, scikit-learn, matplotlib, Pillow, OpenCV).

---

### Step 4: Run the Main Application

Launch the main menu by running from terminal using:

```bash
python ML_Project.py
```

A textual menu will appear, offering options such as:
- Train/Test CNN and SVM models
- Launch GUI for digit drawing and classification
- View plots and saved metrics

---

## Introduction
This project focuses on the classification of handwritten digits using two different machine learning approaches: a Convolutional Neural Network (CNN) and a Support Vector Machine (SVM).

The main objective is to compare the performance of these models on the popular MNIST dataset, as well as on custom user-drawn digits collected via a graphical user interface (GUI). The project includes data preprocessing, model training, evaluation using multiple metrics (accuracy, precision, recall, confusion matrix), and visualizations of results.

Through this comparison, the project highlights the strengths and weaknesses of deep learning versus classical machine learning methods for image classification tasks.

## Dataset

The primary dataset used in this project is the **MNIST dataset** (Modified National Institute of Standards and Technology), which contains **70,000 grayscale images** of handwritten digits ranging from 0 to 9. Each image is **28×28 pixels**, centered and normalized for optimal classification performance.

- **Training set:** 60,000 images  
- **Test set:** 10,000 images

During preprocessing:
- The training set is **split into 90% for training and 10% for development (validation)**.
- Images are flattened and normalized using `StandardScaler` to standardize pixel values.

In addition, the project includes a **custom GUI** allowing users to draw digits on a canvas. These user-generated drawings undergo the same preprocessing steps:
- Centering and cropping the digit
- Resizing to 28×28 pixels
- Normalization with the previously saved scaler

This ensures consistent input formatting between the MNIST dataset and the user-drawn digits.

**MNIST Dataset Source:**  
[https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

---
> **Notes:**  
> - For the **CNN**, the full training set (54,000 images) and development set (6,000 images) were used.  
> - For the **SVM**, only a subset of **10,000 images** was used for training to reduce runtime, as SVMs can be computationally intensive on large datasets.
---

## Model Architectures

### Convolutional Neural Network (CNN)

The CNN model is designed to classify handwritten digits (0-9) from 28x28 grayscale images. Its architecture is as follows:

- **Input**: 28x28 grayscale image (1 channel)
- **Convolutional Layer 1**: 
  - 32 filters, 3x3 kernel, padding=1
  - Activation: ReLU
  - Max Pooling: 2x2
- **Convolutional Layer 2**:
  - 64 filters, 3x3 kernel, padding=1
  - Activation: ReLU
  - Max Pooling: 2x2
- **Flatten**: Converts the output to a vector of size 7x7x64 = 3136
- **Fully Connected Layer 1**:
  - 128 neurons
  - Activation: ReLU
  - Dropout: 25% to prevent overfitting
- **Fully Connected Layer 2 (Output Layer)**:
  - 10 neurons (one per digit class)
  - Activation: Softmax (applied via CrossEntropyLoss)

**Training Details**:
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Early Stopping: Monitored on validation loss with a patience of 5 epochs
- Maximum Epochs: 100

---

### Support Vector Machine (SVM)

The SVM classifier is trained to classify digits using flattened image vectors (28x28 = 784 features). Three different kernels are used:

- **Linear Kernel**
- **Radial Basis Function (RBF) Kernel**
- **Polynomial Kernel**

**Hyperparameter Tuning**:
- Conducted via Grid Search with 5-fold cross-validation.
- Parameter grid:
  - C: [0.1, 1, 10]
  - gamma: ['scale', 0.01, 0.1] (for RBF and Polynomial)
  - degree: [2, 3, 4] (for Polynomial)

Grid Search with 5-fold cross-validation is used to automatically select the best hyperparameters for each kernel, based on cross-validation accuracy computed on the training set.

**Note on 'scale' for gamma**:
The `'scale'` option for the `gamma` parameter in SVM automatically sets `gamma = 1 / (n_features * X.var())`, where `n_features` is the number of features (784 for MNIST) and `X.var()` is the variance of the training data. This helps adapt the gamma value based on the data's distribution, providing a good default for RBF and Polynomial kernels.

**Validation Strategy Difference (CNN vs. SVM)**:
- For **CNN**, the training set is explicitly split into a training subset and a separate development (validation) set. The model is trained on the training subset, and performance is monitored on the validation set for early stopping and hyperparameter tuning.
- For **SVM**, there is no separate validation set. Instead, **5-fold cross-validation** is applied directly on the training data via Grid Search. This ensures that hyperparameter selection (e.g., C, gamma, degree) is robust by averaging

## Project Structure

The project is organized into the following main files and directories:
```
├── cnn_model.py # Defines the architecture of the CNN model
├── train_cnn.py # Handles CNN training, early stopping, and saving plots/metrics 
├── train_svm.py # Handles SVM training with Grid Search and saving models/plots 
├── test_models.py # Evaluates both CNN and SVM models on the test set, saves metrics and confusion matrices 
├── utilities.py # Contains data preprocessing, visualization functions, metrics saving, and confusion matrix plotting 
├── drawing_utils.py # Provides GUI for drawing digits, saving them, and classifying with CNN and SVM models 
├── ML_Project.py # Main entry point: menu for training, testing, visualizing, and using the GUI 
├── requirements.txt # Python dependencies required for running the project 
├── data/ # Contains raw MNIST IDX files (not included in the repository) 
├── data_numpy/ # Contains preprocessed and normalized datasets as NumPy arrays 
├── Models/ # Saved CNN and SVM models 
├── Plots/ # Generated training and evaluation plots (loss, accuracy, confusion matrices) 
├── Metrics/ # Evaluation metrics (accuracy, precision, recall, etc.) saved as text files 
├── UserDrawings/ # User-drawn digit images and corresponding normalized NumPy arrays
```

## Running the project

When you first run the project, it will automatically extract and preprocess the raw MNIST data from the `data/` directory. The processed data will be saved as `.npy` files in the `data_numpy/` directory for faster loading in future runs, eliminating the need to extract and preprocess the dataset each time.

## The Menu
When running `ML_Project.py`, you will be presented with the following menu:

Menu:
1.Train CNN
2.Test CNN
3.Train SVM
4.Test SVM
5.Visualize Data used for CNN
6.Visualize Data used for SVM
7.Evaluate CNN Metrics
8.Evaluate SVM Metrics
9.Display Best SVM Models
10.Drawing Menu
Exit


Each option allows you to perform different tasks within the project:

- **1. Train CNN**: Trains the CNN model on the MNIST dataset, applying early stopping based on validation loss. The best model is saved in the `Models/` directory.

- **2. Test CNN**: Evaluates the trained CNN model on the test set, displays performance metrics (accuracy, precision, recall), saves the confusion matrix, and stores up to three misclassified digit images.

- **3. Train SVM**: Trains three separate SVM models (Linear, RBF, Polynomial) using Grid Search with 5-fold cross-validation to find optimal hyperparameters. The trained models are saved in the `Models/` directory.

- **4. Test SVM**: Evaluates each SVM model on the test set, displays metrics, saves confusion matrices, and stores misclassified examples for each kernel.

- **5. Visualize Data used for CNN**: Displays label distributions and sample images for the training, development, and test sets used by the CNN.

- **6. Visualize Data used for SVM**: Displays label distributions and sample images for the SVM training set (10,000 samples) and test set.

- **7. Evaluate CNN Metrics**: Loads and displays the saved CNN evaluation metrics from `Metrics/CNN/metrics.txt`.

- **8. Evaluate SVM Metrics**: Loads and displays the saved evaluation metrics for all three SVM models from the `Metrics/` directory.

- **9. Display Best SVM Models**: Shows the best hyperparameters (C, gamma, degree) for each SVM kernel, selected during Grid Search.

- **10. Drawing Menu**: Opens an interactive GUI where you can:
  - Draw digits on a canvas.
  - Save the raw and normalized versions of your drawings.
  - Classify your drawings using the trained CNN and SVM models.
  - View saved drawings.

- **0. Exit**: Exits the program.


## The Drawing Menu

Selecting option **10. Drawing Menu** from the main menu opens an interactive interface that allows you to draw your own digit and classify it using the trained CNN and SVM models.

Once inside the Drawing Menu, you will see:
Drawing Menu:
1.Draw and Save Digit
2.Predict from Saved Drawing
3.View Saved Drawings
4.Return to Main Menu


### Options Overview:

- **1. Draw and Save Digit**:  
  Opens a **280x280 pixel canvas** where you can draw a digit using your mouse.  
  After drawing, you can:
  - **Save the drawing** with a custom filename.
  - The system will:
    - Save the **original raw image** (280x280) in `UserDrawings/images/images not normalized/`.
    - **Center, pad, and resize** the digit to **28x28 pixels** (matching MNIST format) and save it as:
      - A normalized image (`images normalized/`).
      - A normalized **NumPy array** (`images_np_data/`), which is used for predictions.

**The Drawing Window:**

<img src="https://github.com/user-attachments/assets/4515783b-0cd4-4c6d-a8f0-045a72188552" alt="drawing window" width="400"/>

- **2. Predict from Saved Drawing**:  
  Allows you to select one of your previously saved drawings (from `images_np_data/`) and classify it using:
  - The trained **CNN** model.
  - The three **SVM** models (Linear, RBF, Polynomial).  
  The predictions for each model are printed in the terminal.

- **3. View Saved Drawings**:  
  Lets you browse and display your saved digit images:
  - Choose between viewing **raw images** (`images not normalized/`) or **normalized images** (`images normalized/`).
  - Select a specific image to display it in a pop-up window.

**Normalized Image (28x28 resized for display):**

<img src="https://github.com/user-attachments/assets/464d6aae-6334-458f-b74c-85779f413efb" alt="FOUR Normalized" width="150"/>

**Non-Normalized Image (280x280):**

<img src="https://github.com/user-attachments/assets/ce12ab30-55c0-4282-ac9f-d7b3f4f20099" alt="FOUR Non-Normalized" width="280"/>

- **0. Return to Main Menu**:  
  Returns to the main project menu.

#### Normalization & Centering Process:
When saving a drawn digit, the system:
1. **Centers the digit**: Crops the bounding box around the digit, pads it back to **280x280**, and recenters it.
2. **Resizes** to **28x28 pixels**.
3. **Normalizes** the pixel values using the same **StandardScaler** used for the MNIST dataset (saved in `Scalers/mnist_scaler.pkl`).

This ensures that your drawn digits are processed in the same way as the MNIST data, improving classification accuracy.

## Evaluation Metrics & Outputs

During training and testing, various outputs are generated to help evaluate the models' performance. These outputs are automatically saved in the following directories:

### 1. Models

- **Directory**: `Models/`
- **Contents**:
  - `cnn_best_model.pth`: The best-performing CNN model, saved after each epoch where validation loss improves.
  - `svm_linear.pkl`, `svm_rbf.pkl`, `svm_poly.pkl`: The best SVM models for each kernel (Linear, RBF, Polynomial), selected via Grid Search with 5-fold cross-validation.

### 2. Plots

- **Directory**: `Plots/`
- **Contents**:
  - `Plots/CNN/`: Contains CNN training and validation plots:
    - `Train Loss and Accuracy over Epochs.png`
    - `Validation Loss and Accuracy over Epochs.png`
    - `confusion_matrix.png`: Confusion matrix of CNN predictions.
  - `Plots/SVM/`: Contains Grid Search accuracy plots for each kernel and confusion matrices:
    - `grid_search_accuracy_linear.png`
    - `grid_search_accuracy_rbf.png`
    - `grid_search_accuracy_poly.png`
    - `confusion_matrix.png` (for each kernel in separate subdirectories).

### 3. Metrics

- **Directory**: `Metrics/`
- **Contents**:
  - `Metrics/CNN/metrics.txt`: Contains the following evaluation metrics for the CNN model:
    - Accuracy
    - Precision (weighted)
    - Recall (weighted)
    - Confusion Matrix
    - Test Time (in seconds)
  - `Metrics/SVM_linear/metrics.txt`, `Metrics/SVM_rbf/metrics.txt`, `Metrics/SVM_poly/metrics.txt`: Contain the same metrics for each SVM kernel.

### 4. Misclassified Examples

- **Directory**: `False predict/`
- **Contents**:
  - `False predict/CNN/`: Up to three misclassified test images from the CNN, saved with filenames indicating the true and predicted labels (e.g., `true_4_pred_9_1.png`).
  - `False predict/SVM/<kernel>/`: Up to three misclassified test images for each SVM kernel.

These outputs provide comprehensive insights into how well the models perform, including accuracy metrics, confusion matrices to identify misclassification patterns, and visual examples of errors.

## Results Summary

| Model     | Accuracy | Precision | Recall | Test Time (s) |
|-----------|----------|-----------|--------|---------------|
| CNN       | 99.03%      | 99.03%       | 99.03%    | 0.263         |
| SVM Linear| 92.16%   | 92.17%    | 92.16% | 9.50         |
| SVM RBF   | 95.94%   | 95.97%    | 95.94% | 27.18         |
| SVM Poly  | 95.02%   | 95.11%    | 95.02% | 18.96        |

The **CNN** model achieved the highest performance across all evaluation metrics:
- **Accuracy**: 99.03%
- **Precision**: 99.03%
- **Recall**: 99.03%
- **Test Time**: 0.263 seconds

The **SVM RBF** kernel achieved the best performance among the SVM models, with:
- **Accuracy**: 95.94%
- **Precision**: 95.97%
- **Recall**: 95.94%
- **Test Time**: 27.18 seconds

The **SVM Polynomial** kernel also performed well, slightly below the RBF kernel but with a shorter test time of 18.96 seconds.

The **SVM Linear** kernel had the lowest accuracy (92.16%) but was the most efficient in terms of test time (9.50 seconds).

Overall, the **CNN** model outperformed all SVM models both in accuracy and in computational efficiency, making it the preferred choice for handwritten digit classification on the MNIST dataset.

Additional evaluation outputs such as confusion matrices, training curves, and misclassified examples are available in the corresponding directories (`Plots/`, `Metrics/`, and `False predict/`).

**confusion matrices**


<p align="center">
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/CNN/confusion_matrix.png?raw=true" alt="CNN Confusion Matrix" width="400"/>
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/SVM/linear/confusion_matrix.png?raw=true" alt="SVM Linear Confusion Matrix" width="400"/>
</p>

<p align="center">
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/SVM/poly/confusion_matrix.png?raw=true" alt="SVM Polynomial Confusion Matrix" width="400"/>
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/SVM/rbf/confusion_matrix.png?raw=true" alt="SVM RBF Confusion Matrix" width="400"/>
</p>

**trainig curves**
**CNN**
<p align="center">
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/CNN/Train Loss and Accuracy over Epochs.png?raw=true" alt="CNN Confusion Matrix" width="400"/>
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/CNN/Validation Loss and Accuracy over Epochs.png?raw=true" alt="SVM Linear Confusion Matrix" width="400"/>
</p>

**trainig curves**
**SVM**
<p align="center">
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/SVM/grid_search_accuracy_linear.png?raw=true" alt="SVM Polynomial Confusion Matrix" width="400"/>
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/SVM/grid_search_accuracy_poly.png?raw=true" alt="SVM RBF Confusion Matrix" width="400"/>
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/Plots/SVM/grid_search_accuracy_rbf.png?raw=true" alt="SVM RBF Confusion Matrix" width="400"/>
  
</p>

**misclassified CNN**
<p align="center">
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/False predict/CNN/true_4_pred_9_1.png?raw=true" alt="CNN Confusion Matrix" width="400"/>
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/False predict/CNN/true_9_pred_4_2.png?raw=true" alt="CNN Confusion Matrix" width="400"/>
  <img src="https://github.com/nirbinyamin/ML_project/blob/main/False predict/CNN/true_6_pred_0_3.png?raw=true" alt="CNN Confusion Matrix" width="400"/>
</p>

**misclassified SVM**
<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/nirbinyamin/ML_project/blob/main/False%20predict/SVM/poly/true_5_pred_4_1.png?raw=true" alt="SVM Polynomial Misclassified" width="400"/><br/>
      <b>Polynomial Kernel</b>
    </td>
    <td align="center">
      <img src="https://github.com/nirbinyamin/ML_project/blob/main/False%20predict/SVM/rbf/true_4_pred_0_2.png?raw=true" alt="SVM RBF Misclassified" width="400"/><br/>
      <b>RBF Kernel</b>
    </td>
    <td align="center">
      <img src="https://github.com/nirbinyamin/ML_project/blob/main/False%20predict/SVM/linear/true_9_pred_4_3.png?raw=true" alt="SVM Linear Misclassified" width="400"/><br/>
      <b>Linear Kernel</b>
    </td>
  </tr>
</table>
