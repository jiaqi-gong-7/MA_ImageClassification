import tensorflow as tf
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras import layers, models   # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # type: ignore

def build_model(input_shape, num_classes):
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model initially

    # Add custom classification layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, train_generator, val_generator, batch_size=32, epochs=10, fine_tune=False, fine_tune_epochs=5):
    # Define the learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=5,
        decay_rate=0.9,
        staircase=False
    )

    # Callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr_schedule), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    print("Training the model with frozen base layers...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Fine-tuning if specified
    if fine_tune:
        print("Fine-tuning the model...")
        # Unfreeze the last few layers of the base model for fine-tuning
        base_model = model.layers[0]  # Get the base ResNet50 model
        for layer in base_model.layers[:-10]:  # Unfreeze the last 10 layers, adjust as necessary
            layer.trainable = False
        for layer in base_model.layers[-10:]:  # Unfreeze the last few layers
            layer.trainable = True

        # Recompile the model with a lower learning rate for fine-tuning
        model.compile(optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        # Continue training with fine-tuning
        history_fine = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            epochs=fine_tune_epochs,
            callbacks=[early_stopping, model_checkpoint]
        )
        return history, history_fine  # Return history for both stages

    return history  # Return history if no fine-tuning


