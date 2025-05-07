import os
from utilities import prepration, load_prepared_data, visualize_data, display_best_svm_models
from train_cnn import train_cnn
from train_svm import train_svm
from test_models import test_cnn, test_svm
from drawing_utils import drawing_menu

def main():
    # Load or prepare data
    data_files = [
        'data_numpy/train_images.npy',
        'data_numpy/train_labels.npy',
        'data_numpy/dev_images.npy',
        'data_numpy/dev_labels.npy',
        'data_numpy/test_images.npy',
        'data_numpy/test_labels.npy'
    ]

    if all(os.path.exists(file) for file in data_files):
        train_imgs, train_lbls, dev_imgs, dev_lbls, test_imgs, test_lbls = load_prepared_data()
        print("Loaded data from data_numpy/")
    else:
        train_imgs, train_lbls, dev_imgs, dev_lbls, test_imgs, test_lbls = prepration()

    while True:
        print("\nMenu:")
        print("1. Train CNN")
        print("2. Test CNN")
        print("3. Train SVM")
        print("4. Test SVM")
        print("5. Visualize Data used for CNN")
        print("6. Visualize Data used for SVM")
        print("7. Evaluate CNN Metrics")
        print("8. Evaluate SVM Metrics")
        print("9. Display Best SVM Models")
        print("10. Drawing Menu")
        print("0. Exit")

        choice = input("Select an option: ")

        if choice == '1':
            train_cnn(train_imgs, train_lbls, dev_imgs, dev_lbls)
        elif choice == '2':
            test_cnn(test_imgs, test_lbls)
        elif choice == '3':
            train_svm(train_imgs, train_lbls)
        elif choice == '4':
            test_svm(test_imgs, test_lbls)
        elif choice == '5':
            visualize_data(train_imgs, train_lbls, dev_imgs, dev_lbls, test_imgs, test_lbls)
        elif choice == '6':
            visualize_data(train_imgs[:10000], train_lbls[:10000], dev_imgs, dev_lbls, test_imgs, test_lbls, isSVM=True)
        elif choice == '7':
            with open("Metrics/CNN/metrics.txt") as f:
                print(f.read())
        elif choice == '8':
            metric_files = [
                "Metrics/SVM_linear/metrics.txt",
                "Metrics/SVM_poly/metrics.txt",
                "Metrics/SVM_rbf/metrics.txt"
            ]
            for file_path in metric_files:
                print(f"\nMetrics from {file_path}:")
                with open(file_path) as f:
                    print(f.read())
        elif choice == '9':
            display_best_svm_models()
        elif choice == '10':
            drawing_menu()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please select again.")

if __name__ == '__main__':
    main()
