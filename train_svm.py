# Import necessary libraries
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# Train SVM models with different kernels using grid search
def train_svm(train_imgs, train_lbls):
    print("Starting SVM model training...")

    # Subsample for faster grid search (can be removed for full dataset)
    train_imgs = train_imgs[:10000]
    train_lbls = train_lbls[:10000]

    # Save raw training data for reproducibility or future analysis
    np.save("data_numpy/train_labels_SVM.npy", train_lbls)
    np.save("data_numpy/train_images_SVM.npy", train_imgs)

    # Flatten each image from 28x28 to 784 features
    train_imgs = train_imgs.reshape(train_imgs.shape[0], -1)
    print("Train set shape:", train_imgs.shape)  # Example: (10000, 784)

    # Calculate variance for gamma estimation and analysis
    variances = np.var(train_imgs, axis=0)            # Variance per feature
    mean_variance = np.mean(variances)                # Mean variance across features
    total_variance = np.var(train_imgs)               # Overall variance of dataset

    # Display variance details
    print(f"Per-feature variances shape: {variances.shape}")  # (784,)
    print(f"Mean Variance across features (axis=0): {mean_variance}")
    print(f"Total Variance (flattened X.var()): {total_variance}")

    # Estimate gamma manually for reference (used when gamma='scale')
    gamma_manual = 1 / (train_imgs.shape[1] * total_variance)
    print(f"Gamma manual (using total variance): {gamma_manual}")

    # Define kernel types to train on
    kernels = ['rbf', 'poly', 'linear']

    # Ensure directories exist for saving models and plots
    os.makedirs("Models", exist_ok=True)
    os.makedirs("Plots/SVM", exist_ok=True)

    # Loop over each kernel type and perform grid search
    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel...")

        # Define parameter grid for each kernel
        if kernel == 'linear':
            param_grid = {'C': [0.1, 1, 10]}  # Regularization only
        elif kernel == 'rbf':
            param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1]}
        elif kernel == 'poly':
            param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1], 'degree': [2, 3, 4]}

        # Initialize the base SVM model
        svm = SVC(kernel=kernel)

        # Perform grid search with 5-fold cross-validation
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, n_jobs=1, scoring='accuracy', refit=True, verbose=3
        )

        print("Running Grid Search...")
        grid_search.fit(train_imgs, train_lbls)
        print("Grid Search completed.")

        # Display the best parameters found
        print("Best parameters found:", grid_search.best_params_)

        best_model = grid_search.best_estimator_

        # For kernels that use gamma, print the final computed value
        if kernel in ['rbf', 'poly']:
            print(f"\nComputed gamma for {kernel}: {best_model._gamma}")

        # Save the best model to disk
        joblib.dump(best_model, f"Models/svm_{kernel}.pkl")
        print(f"Best {kernel} model saved at Models/svm_{kernel}.pkl")

        # Visualize grid search scores
        results = grid_search.cv_results_
        scores = np.array(results['mean_test_score'])

        plt.figure()
        plt.plot(scores, 'o')  # Plot scores per parameter set
        plt.title(f"Grid Search Accuracy for {kernel} Kernel")
        plt.xlabel("Parameter Set Index")
        plt.ylabel("Mean CV Accuracy")
        plt.savefig(f"Plots/SVM/grid_search_accuracy_{kernel}.png")
        plt.close()

        print(f"Training plots for {kernel} saved at Plots/SVM/")

    print("\nTraining for all SVM kernels completed!")
