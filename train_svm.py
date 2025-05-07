from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

def train_svm(train_imgs, train_lbls):
    print("Starting SVM model training...")
    train_imgs = train_imgs[:10000]
    train_lbls = train_lbls[:10000]

    np.save("data_numpy/train_labels_SVM.npy", train_lbls)
    np.save("data_numpy/train_images_SVM.npy", train_imgs)
    train_imgs = train_imgs.reshape(train_imgs.shape[0], -1)

    print("Train set shape:", train_imgs.shape)  # Example: (100, 784)

    # Compute variance per feature (axis=0)
    variances = np.var(train_imgs, axis=0)
    mean_variance = np.mean(variances)

    # Compute total variance (over all elements)
    total_variance = np.var(train_imgs)

    # Print detailed variance information
    print(f"Per-feature variances shape: {variances.shape}")  # (784,)
    print(f"Mean Variance across features (axis=0): {mean_variance}")
    print(f"Total Variance (flattened X.var()): {total_variance}")

    # Compute gamma manually for 'scale'
    gamma_manual = 1 / (train_imgs.shape[1] * total_variance)
    print(f"Gamma manual (using total variance): {gamma_manual}")

    kernels = ['rbf', 'poly', 'linear']
   

    os.makedirs("Models", exist_ok=True)
    os.makedirs("Plots/SVM", exist_ok=True)

    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel...")
        if kernel == 'linear':
            param_grid = {'C': [0.1, 1, 10]}
        elif kernel == 'rbf':
            param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1]}  
        elif kernel == 'poly':
            param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1], 'degree': [2, 3, 4]}

        svm = SVC(kernel=kernel)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, n_jobs=1, scoring='accuracy', refit=True, verbose=3
        )

        print("Running Grid Search...")
        grid_search.fit(train_imgs, train_lbls)
        print("Grid Search completed.")

        print("Best parameters found:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        if kernel in ['rbf', 'poly']:
            print(f"\nComputed gamma for {kernel}: {best_model._gamma}")

        joblib.dump(best_model, f"Models/svm_{kernel}.pkl")
        print(f"Best {kernel} model saved at Models/svm_{kernel}.pkl")

        # Plot the scores
        results = grid_search.cv_results_
        scores = np.array(results['mean_test_score'])

        plt.figure()
        plt.plot(scores, 'o')
        plt.title(f"Grid Search Accuracy for {kernel} Kernel")
        plt.xlabel("Parameter Set Index")
        plt.ylabel("Mean CV Accuracy")
        plt.savefig(f"Plots/SVM/grid_search_accuracy_{kernel}.png")
        plt.close()

        print(f"Training plots for {kernel} saved at Plots/SVM/")

    print("\nTraining for all SVM kernels completed!")

