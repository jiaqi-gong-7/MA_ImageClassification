import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import os
import json
import sys 

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def load_and_preprocess_image(img_path, target_size=(300, 300)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img

def predict_and_visualize():
    #print(sys.argv[1])
    img_path = sys.argv[1]
    img_tensor, original_img = load_and_preprocess_image(img_path)
    prediction = model.predict(img_tensor)[0]
    class_index = np.argmax(prediction)
    class_name = CLASS_NAMES[class_index]
    confidence = float(prediction[class_index])
    color, quality = QUALITY_LABELS.get(class_name, ('ORANGE', 'Possibly Defective'))

    # === JSON Output ===
    result = {
        "prediction_vector": {
            CLASS_NAMES[i]: round(float(prob), 4) for i, prob in enumerate(prediction)
        },
        "final_prediction": {
            "label": class_name,
            "confidence": round(confidence, 4),
            "quality": quality,
            "color": color
        }
    }
    print(json.dumps(result, indent=2))

    # === Image display and saving ===
    fig = plt.figure(figsize=(8, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 5, 2])

    ax0 = fig.add_subplot(gs[0])
    ax0.axis('off')
    ax0.text(
        0.5, 0.5,
        f"Prediction: {class_name}\nConfidence: {confidence:.2f}\nQuality: {quality} [{color}]",
        fontsize=15, ha='center', va='center', color=color.lower()
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
    save_path = "results/reports/single_prediction_result.pdf"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()

    # === Human-in-the-loop correction  ===
    print(f"\nModel predicted '{class_name}' with confidence {confidence:.2f}.")
    user_feedback = input("Is the prediction correct? (y/n): ").strip().lower()

    if user_feedback == 'n':
        print("\nAvailable classes:")
        for i, label in enumerate(CLASS_NAMES):
            print(f"{i}: {label}")
        correct_index = input("Enter the correct class index (0-4): ").strip()
        try:
            correct_index = int(correct_index)
            if 0 <= correct_index < len(CLASS_NAMES):
                correct_label = CLASS_NAMES[correct_index]
                correction_dir = os.path.join("corrections", correct_label)
                os.makedirs(correction_dir, exist_ok=True)
                img_name = os.path.basename(img_path)
                save_img_path = os.path.join(correction_dir, img_name)
                original_img.save(save_img_path)
                print(f"Misclassified image saved to: {save_img_path}")
            else:
                print("Invalid class index.")
        except:
            print("Invalid input. Correction skipped.")

# Example call
predict_and_visualize()
