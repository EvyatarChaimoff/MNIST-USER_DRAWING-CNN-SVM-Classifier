import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageDraw
import numpy as np
import os
import joblib
import torch
import matplotlib.pyplot as plt
from cnn_model import CNNModel

# Constants
CANVAS_SIZE = 280
IMAGE_SIZE = 28
SAVE_DIR = "UserDrawings"
SCALER_PATH = "Scalers/mnist_scaler.pkl"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the StandardScaler
scaler = joblib.load(SCALER_PATH)

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Digit")
        
        self.canvas = tk.Canvas(self.master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.pack()
        
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()

        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)
        
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)
        
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)

    def save_image(self):
        # Directories
        images_dir = os.path.join(SAVE_DIR, "images")
        not_norm_dir = os.path.join(images_dir, "images not normalized")
        norm_dir = os.path.join(images_dir, "images normalized")
        np_data_dir = os.path.join(SAVE_DIR, "images_np_data")

        os.makedirs(not_norm_dir, exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)
        os.makedirs(np_data_dir, exist_ok=True)

        # Original image (280x280), convert to numpy
        img_array = np.array(self.image)
        # Invert: white background to black background

        # Prompt for filename
        filename = simpledialog.askstring("Save As", "Enter file name (without extension):")
        if filename:
            # Save the original, non-normalized, non-centered image (280x280)
            Image.fromarray(img_array).save(os.path.join(not_norm_dir, f"{filename}.png"))
            print(f"Saved not normalized image at {not_norm_dir}/{filename}.png")

            # Center, pad back to 280x280, resize to 28x28
            img_centered = center_and_resize(img_array)

            # Normalize using StandardScaler
            img_flat = img_centered.reshape(1, -1)
            img_scaled = scaler.transform(img_flat).reshape(IMAGE_SIZE, IMAGE_SIZE)

            # Save normalized image (centered)
            plt.imsave(os.path.join(norm_dir, f"{filename}.png"), img_scaled, cmap='gray')
            print(f"Saved normalized image at {norm_dir}/{filename}.png")

            # Save normalized numpy array
            np.save(os.path.join(np_data_dir, f"{filename}.npy"), img_scaled)
            print(f"Saved numpy array at {np_data_dir}/{filename}.npy")


      
def predict_drawing(file_path):
    # Load the image
    img_array = np.load(file_path)

    # Prepare for CNN
    cnn_input = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CNN model
    cnn_model = CNNModel().to(device)
    cnn_model.load_state_dict(torch.load("Models/cnn_best_model.pth", map_location=device))
    cnn_model.eval()

    with torch.no_grad():
        output = cnn_model(cnn_input.to(device))
        prediction = torch.argmax(output, dim=1).item()
        print(f"CNN predicts: {prediction}")

    # Prepare for SVM
    svm_input = img_array.reshape(1, -1)
    kernels = ['linear', 'rbf', 'poly']

    for kernel in kernels:
        svm_model = joblib.load(f"Models/svm_{kernel}.pkl")
        svm_prediction = svm_model.predict(svm_input)[0]
        print(f"SVM ({kernel}) predicts: {svm_prediction}")

def delete_drawing(filename_no_ext):
    paths = [
        os.path.join(SAVE_DIR, "images_np_data", f"{filename_no_ext}.npy"),
        os.path.join(SAVE_DIR, "images", "images normalized", f"{filename_no_ext}.png"),
        os.path.join(SAVE_DIR, "images", "images not normalized", f"{filename_no_ext}.png")
    ]
    
    deleted_any = False
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted: {path}")
            deleted_any = True
        else:
            print(f"Not found: {path}")
    if not deleted_any:
        print("No matching files were found to delete.")

def drawing_menu():
    while True:
        print("\nDrawing Menu:")
        print("1. Draw and Save Digit")
        print("2. Predict from Saved Drawing")
        print("3. View Saved Drawings")
        print("4. Delete a Saved Drawing")
        print("0. Return to Main Menu")

        choice = input("Select an option: ")

        if choice == '1':
            root = tk.Tk()
            app = DrawingApp(root)
            root.mainloop()
        elif choice == '2':
            np_data_dir = os.path.join(SAVE_DIR, "images_np_data")
            files = [f for f in os.listdir(np_data_dir) if f.endswith(".npy")]
            if not files:
                print("No drawings found in UserDrawings/images_np_data/.")
                continue
            print("Available drawings:")
            for idx, file in enumerate(files):
                print(f"{idx + 1}. {file}")
            file_idx = int(input("Select file number: ")) - 1
            if 0 <= file_idx < len(files):
                predict_drawing(os.path.join(np_data_dir, files[file_idx]))
            else:
                print("Invalid selection.")

        elif choice == '3':
            # View Saved Drawings
            print("\nSelect image type to view:")
            print("1. Images Not Normalized")
            print("2. Images Normalized")
            img_choice = input("Select an option: ")

            if img_choice == '1':
                view_dir = os.path.join(SAVE_DIR, "images", "images not normalized")
            elif img_choice == '2':
                view_dir = os.path.join(SAVE_DIR, "images", "images normalized")
            else:
                print("Invalid option.")
                continue

            files = [f for f in os.listdir(view_dir) if f.endswith(".png")]
            if not files:
                print("No images found.")
                continue

            print("\nAvailable images:")
            for idx, file in enumerate(files):
                print(f"{idx + 1}. {file}")
            file_idx = int(input("Select file number: ")) - 1
            if 0 <= file_idx < len(files):
                img_path = os.path.join(view_dir, files[file_idx])
                img = Image.open(img_path)
                plt.figure()
                plt.imshow(img, cmap='gray')
                plt.title(f"Viewing: {files[file_idx]}")
                plt.axis('off')
                plt.show()
            else:
                print("Invalid selection.")
        
        elif choice == '4':
            np_data_dir = os.path.join(SAVE_DIR, "images_np_data")
            files = [f for f in os.listdir(np_data_dir) if f.endswith(".npy")]
            if not files:
                print("No drawings found in UserDrawings/images_np_data/.")
                continue
            print("Available drawings:")
            for idx, file in enumerate(files):
                print(f"{idx + 1}. {file}")
            try:
                file_idx = int(input("Select file number to delete: ")) - 1
                if 0 <= file_idx < len(files):
                    filename_no_ext = os.path.splitext(files[file_idx])[0]
                    confirm = input(f"Are you sure you want to delete '{filename_no_ext}'? (y/n): ")
                    if confirm.lower() == 'y':
                        delete_drawing(filename_no_ext)
                    else:
                        print("Deletion canceled.")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == '0':
            break
        else:
            print("Invalid option. Please select again.")

def center_and_resize(img_array, canvas_size=280, target_size=28):
    # Initialize bounds
    top, bottom = None, None
    left, right = None, None

    # Search rows (top to bottom)
    for r in range(img_array.shape[0]):
        if np.any(img_array[r, :] < 255):
            top = r
            break

    # Search rows (bottom to top)
    for r in range(img_array.shape[0]-1, -1, -1):
        if np.any(img_array[r, :] < 255):
            bottom = r
            break

    # Search columns (left to right)
    for c in range(img_array.shape[1]):
        if np.any(img_array[:, c] < 255):
            left = c
            break

    # Search columns (right to left)
    for c in range(img_array.shape[1]-1, -1, -1):
        if np.any(img_array[:, c] < 255):
            right = c
            break

    # Check if anything found
    if top is None or bottom is None or left is None or right is None:
        print("No digit found.")
        return np.zeros((target_size, target_size), dtype=np.uint8)

    
    # Crop
    img_cropped = img_array[top:bottom+1, left:right+1]
    img_cropped_inverted = 255 - img_cropped
    
    img_pil_cropped = Image.fromarray(img_cropped_inverted)
    
    # Pad to canvas_size
    width, height = img_pil_cropped.size
    canvas = Image.new('L', (canvas_size, canvas_size), color=0)
    upper_left = ((canvas_size - width) // 2, (canvas_size - height) // 2)
    canvas.paste(img_pil_cropped, upper_left)

    # Save padded debug
    canvas.save("debug_padded.png")

    # Resize to target_size
    canvas_resized = canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)

    return np.array(canvas_resized)





