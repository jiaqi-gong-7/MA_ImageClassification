import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint # type: ignore
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns 
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths to your dataset
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Step 1: Data Augmentation and Image Generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values between 0 and 1
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle = False
)

# Step 2: Load the Pre-trained Model (ResNet50 in this case)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Step 3: Add Custom Classification Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), # 全局平均池化层， 减少参数数量
    layers.BatchNormalization(),     # 正则化层， 稳定训练过程
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),             # Dropout, 防止过拟合
    layers.BatchNormalization(),
    layers.Dense(train_generator.num_classes, activation='softmax')  # 输出层，Number of classes from train_generator
])

# Step 4: Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the Model
# 动态调整学习率的调度器
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr*tf.math.exp(-0.1)

# 早停机制与模型检查点
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=10,  # Adjust based on performance
    callbacks=[LearningRateScheduler(scheduler), early_stopping, model_checkpoint]
    )

# Optional Step 6: Fine-tuning (Unfreeze some base model layers)
# Unfreeze the last few layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Unfreeze last 10 layers, you can adjust this number
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(1e-5),  # Lower learning rate for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=5, # A few more epochs for fine-tuning
    callbacks=[early_stopping, model_checkpoint]
)

# Step 7: Evaluate the Model on the Test Set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test accuracy:", test_accuracy)

# Generate predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys())))

# Step 8: Automated Report Generation
def generate_report(y_true, y_pred_classes):
    total = len(y_true)
    correct = np.sum(y_true == y_pred_classes)
    accuracy = correct / total * 100
    defective_indices = np.where(y_true != y_pred_classes)[0]

    report_content = f"""
    Total Number of Samples: {total}
    Number of Correctly Classified Samples: {correct}
    Number of Misclassified Samples: {len(defective_indices)}
    Accuracy: {accuracy:.2f}%
    Indices of Misclassified Samples: {defective_indices.tolist()}
    """
    with open('test_report.txt', 'w') as f:
        f.write(report_content)

generate_report(y_true, y_pred_classes)
