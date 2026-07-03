import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. LOAD YOUR TRAINED MODEL AND CASCADES ---
try:
    model = load_model('eye_classifier.keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')

# --- 2. SETUP VARIABLES ---
IMG_SIZE = (80, 80)
# The model outputs 0 for 'closed_eyes' and 1 for 'open_eyes'
CLASS_LABELS = ['CLOSED', 'OPEN'] 

# Drowsiness detection parameters
CLOSED_EYES_COUNTER = 0
CLOSED_EYES_THRESH = 10 # Number of consecutive frames with closed eyes to trigger an alert

# --- 3. LIVE DETECTION LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Assume eyes are open unless we find a closed one
        is_eyes_closed = True 

        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                
                # Preprocess the eye ROI for the model
                eye_resized = cv2.resize(eye_roi, IMG_SIZE)
                eye_input = np.expand_dims(eye_resized, axis=0)

                # Make a prediction
                prediction = model.predict(eye_input, verbose=0)
                
                # Get the class label (0 or 1)
                predicted_class = 1 if prediction[0][0] > 0.5 else 0

                # If ANY detected eye is open, we consider the eyes open
                if CLASS_LABELS[predicted_class] == 'OPEN':
                    is_eyes_closed = False
                    break # No need to check other eyes
        
        # --- DROWSINESS LOGIC ---
        if is_eyes_closed:
            CLOSED_EYES_COUNTER += 1
            status_text = f"EYES CLOSED ({CLOSED_EYES_COUNTER})"
            status_color = (0, 0, 255) # Red
        else:
            CLOSED_EYES_COUNTER = 0 # Reset counter if eyes are open
            status_text = "EYES OPEN"
            status_color = (0, 255, 0) # Green

        # Display the status on the frame
        cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Trigger alert if threshold is met
        if CLOSED_EYES_COUNTER >= CLOSED_EYES_THRESH:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the final frame
    cv2.imshow('Drowsiness Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()