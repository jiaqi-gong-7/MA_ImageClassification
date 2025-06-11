import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, ResNet50, DenseNet121, EfficientNetB0  # type: ignore
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts # type: ignore
from sklearn.utils.class_weight import compute_class_weight 
import numpy as np 
import matplotlib.pyplot as plt
import os

# # Enable mixed precision & XLA acceleration
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)
AUTOTUNE = tf.data.AUTOTUNE

# Setting the random seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_random_seed(42)


# Building model
def build_model(input_shape, num_classes):
    #base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    #base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,

        # Add more convolutional layers for feature extraction
        layers.Conv2D(512, (1, 1), use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(256, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),  # Increase the Dropout ratio to prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history, stage='initial'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.title(f'{stage.capitalize()} Training Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title(f'{stage.capitalize()} Training Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs('results/reports', exist_ok=True)
    plt.savefig(f'results/reports/training_curves_{stage}.png')
    plt.close()


# Training and fine-tuning
def train_model(model, train_generator, val_generator, num_train_samples, num_val_samples, 
                batch_size=32, epochs=20, fine_tune=True, fine_tune_epochs=10):
    
    steps_per_epoch = num_train_samples // batch_size + int(num_train_samples % batch_size != 0)
    validation_steps = num_val_samples // batch_size + int(num_val_samples % batch_size != 0)
    
    lr_schedule = CosineDecayRestarts(initial_learning_rate=0.0003, first_decay_steps=steps_per_epoch * 2)
    optimizer = Adam(learning_rate=lr_schedule)

    loss_fn = 'categorical_crossentropy'  
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, save_format='keras')
    model_checkpoint = ModelCheckpoint(
    filepath='best_model_ResNet50.h5',  
    #filepath='best_model_EfficientNetB3_1.h5',
    monitor='val_loss',
    save_best_only=True
)

    y_train_full = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in train_generator])
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_full), y=y_train_full)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Initial training
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weights_dict
    )

    plot_training_history(history, stage='initial')

    # Fine-tuning the model
    if fine_tune:
        print("Fine-tuning the model...")
        
        # Thaw some layers gradually
        for layer in model.layers[0].layers[:-200]:  # Unfreeze only the last 200 layers
            layer.trainable = False
        for layer in model.layers[0].layers[-200:]:
            layer.trainable = True

        fine_tune_optimizer = Adam(learning_rate=1e-5)
        
        model.compile(optimizer=fine_tune_optimizer, loss=loss_fn, metrics=['accuracy'])

        history_fine = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=fine_tune_epochs,
            callbacks=[early_stopping, model_checkpoint],
            class_weight=class_weights_dict
        )

        plot_training_history(history_fine, stage='finetune')
        return {'initial_training': history, 'fine_tuning': history_fine}

    return {'initial_training': history}

MODEL_PATH = "best_model_EfficientNetB3_1.h5"
def save_model(model):
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")