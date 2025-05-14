import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageEnhance

# 模型路径
MODEL_PATH = 'best_model_ResNet50_1.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# 类别与质量映射
CLASS_NAMES = ['Complete product', 'Missing CD', 'No ropes', 'No strain relief', 'Valid pcb']
QUALITY_LABELS = {
    'Complete product': ('GREEN', 'Qualified Product'),
    'Valid pcb': ('GREEN', 'Qualified Product'),
    'Missing CD': ('RED', 'Defective Product'),
    'No ropes': ('RED', 'Defective Product'),
    'No strain relief': ('RED', 'Defective Product')
}

# 加载并预处理图像
def load_and_preprocess_image(img_path, target_size=(300, 300)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)

    # 图像增强（可选）
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 使用 ResNet50 的预处理方式（标准化为 [-123.68, -116.78, -103.94] 的均值）
    img_array = preprocess_input(img_array)

    return img_array, img

# 主函数
def predict_image_quality(img_path):
    img_tensor, original_img = load_and_preprocess_image(img_path)
    prediction = model.predict(img_tensor)[0]
    class_index = np.argmax(prediction)
    class_name = CLASS_NAMES[class_index]
    confidence = prediction[class_index]
    color, quality = QUALITY_LABELS.get(class_name, ('ORANGE', 'Possibly Defective'))

    print("\n=== Softmax Prediction Vector ===")
    for i, prob in enumerate(prediction):
        print(f"{CLASS_NAMES[i]:<18}: {prob:.4f}")
    print(f"\nFinal Prediction: {class_name} ({confidence:.2f}), Quality: {quality} [{color}]")

    # 图像结果可视化
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
    save_path = "results/reports/single_prediction_result.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.show()

    print(f"Prediction result saved to: {save_path}")

# 示例调用
predict_image_quality("data/test_samples/img_003.jpg")
