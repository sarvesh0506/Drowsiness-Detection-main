import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import os

# --- 1. DATA LOADING AND PREPARATION ---
BATCH_SIZE = 32
IMG_SIZE = (80, 80) # A good size for eye images

# Load datasets from your specific directories
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/trains',  # <--- CHANGED
    labels='inferred',
    label_mode='binary', # 0 for 'closed_eyes', 1 for 'open_eyes'
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/vals',    # <--- CHANGED
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/tests',   # <--- CHANGED
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Class names:", train_dataset.class_names) # Should print ['closed_eyes', 'open_eyes']

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 2. DATA AUGMENTATION AND MODEL BUILDING ---

# Create a simple data augmentation layer
data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
])

# Define the input shape
IMG_SHAPE = IMG_SIZE + (3,)

# Load the base model (MobileNetV2) without its top classification layer
base_model = MobileNetV2(input_shape=IMG_SHAPE,
                         include_top=False,
                         weights='imagenet')

# Freeze the base model so we only train our new layers
base_model.trainable = False

# Create our new model on top
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs) # Apply augmentation
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # Preprocess for MobileNetV2
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) # Add dropout for regularization
outputs = Dense(1, activation='sigmoid')(x) # Single output neuron for binary classification

model = Model(inputs, outputs)

# --- 3. COMPILE THE MODEL ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. TRAIN THE MODEL ---
print("\n--- Starting Training ---")
EPOCHS = 15
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)

# --- 5. EVALUATE AND SAVE ---
print("\n--- Evaluating on Test Data ---")
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Accuracy: {accuracy:.2f}')

print("\n--- Saving model to eye_classifier.keras ---")
model.save('eye_classifier.keras')
print("Model saved successfully!")