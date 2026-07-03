import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# --- 1. DEFINE CUSTOM LOSS FUNCTION (Required for loading the model) ---
# This function must be defined so TensorFlow knows how to load the model.
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]
    h_pred = yhat[:,3] - yhat[:,1]
    w_pred = yhat[:,2] - yhat[:,0]
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    return delta_coord + delta_size

# --- 2. SETUP GPU ---
# Configure GPU for memory growth (optional but good practice).
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# --- 3. LOAD THE TRAINED MODEL ---
try:
    # Load the model with the custom_objects argument.
    facetracker = load_model('facetracker_mobilenet.h5',
                             custom_objects={'localization_loss': localization_loss})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'facetracker_mobilenet.h5' is in the same folder as this script.")
    exit()

# --- 4. START THE LIVE DETECTION ---
# Use index 0 for the default built-in camera.
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Get the height and width of the frame for scaling the bounding box.
    frame_h, frame_w, _ = frame.shape
    
    # Preprocess the frame for the model.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    
    # Make a prediction. `verbose=0` hides the progress bar for a cleaner output.
    yhat = facetracker.predict(np.expand_dims(resized / 255, 0), verbose=0)
    sample_coords = yhat[1][0]
    
    # Draw the results on the original frame.
    if yhat[0] > 0.5: # If confidence is over 50%.
        # Scale the normalized coordinates to the original frame size.
        x1 = int(sample_coords[0] * frame_w)
        y1 = int(sample_coords[1] * frame_h)
        x2 = int(sample_coords[2] * frame_w)
        y2 = int(sample_coords[3] * frame_h)
        
        # Draw the main bounding box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw the label background.
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + 80, y1), (255, 0, 0), -1)
        
        # Put the "face" text on the label.
        cv2.putText(frame, 'face', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()