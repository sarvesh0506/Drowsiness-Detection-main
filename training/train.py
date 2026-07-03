import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
import albumentations as A

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Albumentations augmentation pipeline
aug_pipeline = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.HorizontalFlip(p=0.5),
])

def albumentations_preprocess(img):
    """Bridge function to apply Albumentations transformations inside Keras ImageDataGenerator."""
    # Keras inputs are normalized float32 images in range [0, 1] or raw [0, 255] depending on rescale config.
    # We expect raw uint8 image array.
    img_uint8 = (img * 255.0).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    augmented = aug_pipeline(image=img_uint8)
    # Re-normalize to [0, 1]
    return augmented['image'].astype(np.float32) / 255.0

def build_model(input_shape=(64, 64, 3)):
    """Define the CNN structure for binary classification (0: Open, 1: Closed)."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Classification Head
        Flatten(),
        Dense(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Sigmoid for binary output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    base_dir = 'data'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    # 1. Verify dataset presence, run setup if missing
    if not os.path.exists(train_dir) or len(os.listdir(os.path.join(train_dir, 'open_eyes'))) == 0:
        logger.info("Dataset directories empty. Invoking dataset_prep.py...")
        from dataset_prep import create_directory_structure, build_mock_dataset
        create_directory_structure(base_dir)
        build_mock_dataset(base_dir)

    # 2. Setup Data Generators
    # Train generator applies Albumentations augmentations
    train_datagen = ImageDataGenerator(
        preprocessing_function=albumentations_preprocess
    )
    
    # Validation generator only normalizes pixel range
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    img_size = (64, 64)
    batch_size = 16
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['open_eyes', 'closed_eyes'], # Open maps to 0, Closed maps to 1
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['open_eyes', 'closed_eyes'],
        shuffle=False
    )
    
    # 3. Create model and print summary
    model = build_model(input_shape=(img_size[0], img_size[1], 3))
    model.summary()
    
    # Ensure output folders exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    
    # 4. Setup Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='models/eye_classifier.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1
        )
    ]
    
    # 5. Model Training
    epochs = 20
    logger.info("Starting CNN Eye Classifier training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Explicitly save final weights if checkpoints missed
    model.save('models/eye_classifier.h5')
    logger.info("Model saved to models/eye_classifier.h5")
    
    # 6. Generate loss/accuracy figures
    plot_training_results(history)

def plot_training_results(history):
    """Plot metrics and save as static images for publication evaluation."""
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy', color='teal')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Train Loss', color='teal')
    plt.plot(epochs_range, history.history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = 'training/accuracy_loss_metrics.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    logger.info(f"Loss/Accuracy curves saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    main()
