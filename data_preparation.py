import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def create_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
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

    # Normalization for validation and test sets
    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Generators for train, validation, and test sets
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )
    test_generator = val_test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    
    return train_generator, val_generator, test_generator
