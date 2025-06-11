import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import matplotlib.pyplot as plt

# Model Path
MODEL_PATH = 'best_model_ResNet50_1.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Category and Quality Mapping
CLASS_NAMES = ['Complete product', 'Missing CD', 'No ropes', 'No strain relief', 'Valid pcb']
QUALITY_LABELS = {
    'Complete product': ('GREEN', 'Qualified Product'),
    'Valid pcb': ('GREEN', 'Qualified Product'),
    'Missing CD': ('RED', 'Defective Product'),
    'No ropes': ('RED', 'Defective Product'),
    'No strain relief': ('RED', 'Defective Product')
}

# Load and preprocess the image
def load_and_preprocess_image(img, target_size=(300, 300)):
    img = img.resize(target_size)

    # Image enhancement 
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Use ResNet50 preprocessing (normalized to the mean of [-123.68, -116.78, -103.94])
    img_array = preprocess_input(img_array)

    return img_array, img

# Main function - predicting image quality
def predict_image_quality_from_camera(frame):
    img_tensor, original_img = load_and_preprocess_image(frame)
    prediction = model.predict(img_tensor)[0]
    class_index = np.argmax(prediction)
    class_name = CLASS_NAMES[class_index]
    confidence = prediction[class_index]
    color, quality = QUALITY_LABELS.get(class_name, ('ORANGE', 'Possibly Defective'))

    print("\n=== Softmax Prediction Vector ===")
    for i, prob in enumerate(prediction):
        print(f"{CLASS_NAMES[i]:<18}: {prob:.4f}")
    print(f"\nFinal Prediction: {class_name} ({confidence:.2f}), Quality: {quality} [{color}]")

    # Image results visualization
    fig = plt.figure(figsize=(8, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 5, 2])

    ax0 = fig.add_subplot(gs[0])
    ax0.axis('off')
    ax0.text(
        0.5, 0.5,
        f"Prediction: {class_name}\nConfidence: {confidence:.2f}\nQuality: {quality} [{color}]",
        fontsize=15, ha='center', va='center', color=color.lower(), wrap=True
    )

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(original_img)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[2])
    bars = ax2.bar(CLASS_NAMES, prediction, color='lightgray')
    bars[class_index].set_color(color.lower())
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Probability")
    ax2.set_title("Softmax Confidence for All Classes", fontsize=12)
    for i, v in enumerate(prediction):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

    os.makedirs("results/reports", exist_ok=True)
    save_path = "results/reports/camera_prediction_result.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.show()

    print(f"Prediction result saved to: {save_path}")

    # Displaying results in the GUI
    result_label.config(text=f"Prediction: {class_name}\nConfidence: {confidence:.2f}\nQuality: {quality} [{color}]")
    result_img = ImageTk.PhotoImage(original_img)
    img_label.config(image=result_img)
    img_label.image = result_img

# Camera capture function
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return

    print("Camera is open. Press 'Capture' to take a photo.")
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        # Display live video stream
        cv2.imshow("Real-time Camera", frame)

        #  Update the GUI
        root.update()

        if capture_button_clicked:  # Trigger photo taking
            print("Capturing image...")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            predict_image_quality_from_camera(pil_image)

        elif quit_button_clicked:  # Exit the program
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI initialization
root = tk.Tk()
root.title("AI Camera Prediction")
root.geometry("600x800")

# Camera image display frame
img_label = tk.Label(root)
img_label.pack()

# Prediction result display box
result_label = tk.Label(root, text="Prediction: None\nConfidence: 0.00\nQuality: None", font=("Helvetica", 14))
result_label.pack()

# Button
capture_button_clicked = False
quit_button_clicked = False

def on_capture_button_click():
    global capture_button_clicked
    capture_button_clicked = True
    capture_image_from_camera()

def on_quit_button_click():
    global quit_button_clicked
    quit_button_clicked = True
    root.quit()

capture_button = tk.Button(root, text="Capture", command=on_capture_button_click)
capture_button.pack(pady=20)

quit_button = tk.Button(root, text="Quit", command=on_quit_button_click)
quit_button.pack(pady=20)

# Launch the GUI
root.after(1, capture_image_from_camera)  # Start the camera thread

root.mainloop()
