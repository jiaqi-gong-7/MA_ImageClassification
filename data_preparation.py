import tensorflow as tf
import os
#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input


def count_images(directory):
    total_images = sum([len(files) for _, _, files in os.walk(directory)])
    return total_images


def create_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):

    # ✅ 计算图像数量
    num_train_samples = count_images(train_dir)
    num_val_samples = count_images(val_dir)
    num_test_samples = count_images(test_dir)

    # ✅ 加载原始图像数据集（不归一化）
    train_generator = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )

    class_names = train_generator.class_names
    num_classes = len(class_names)

    # ✅ 数据增强层
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=(*img_size, 3)),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.GaussianNoise(0.05),
    ])

    # ✅ 使用 ResNet50/DenseNet121 的预处理方式：从 [0, 255] → [-1, 1]
    #normalization_layer = tf.keras.layers.Lambda(preprocess_input)

    # ✅ EfficientNet 的预处理层（归一化 [0, 255] -> [0, 1]）
    normalization_layer = tf.keras.layers.Lambda(preprocess_input)


    # ✅ 应用增强与归一化
    train_generator = train_generator.map(
        lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # ✅ 验证集和测试集归一化
    val_generator = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    val_generator = val_generator.map(lambda x, y: (normalization_layer(x), y))

    test_generator = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    test_generator = test_generator.map(lambda x, y: (normalization_layer(x), y))

    # ✅ 加速预取
    train_generator = train_generator.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_generator = val_generator.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_generator = test_generator.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Data preparation completed successfully!")
    print(f"Number of Training Samples: {num_train_samples}")
    print(f"Number of Validation Samples: {num_val_samples}")
    print(f"Number of Test Samples: {num_test_samples}")

    return train_generator, val_generator, test_generator, class_names, num_classes, num_train_samples, num_val_samples
