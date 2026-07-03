import cv2
import os
import uuid

# --- 1. SETUP PATHS AND CASCADES ---
OPEN_EYES_PATH = os.path.join('data', 'open_eyes')
CLOSED_EYES_PATH = os.path.join('data', 'closed_eyes')
os.makedirs(OPEN_EYES_PATH, exist_ok=True)
os.makedirs(CLOSED_EYES_PATH, exist_ok=True)

# Load the Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')

cap = cv2.VideoCapture(0)

# --- 2. MAIN COLLECTION LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Create a region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Process and save the detected eyes
        eye_rois_to_save = []
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye for feedback
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            if eye_roi.size != 0:
                eye_rois_to_save.append(eye_roi)
    
    cv2.imshow('Data Collection', frame)
    
    # --- 3. KEY PRESS LOGIC ---
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'o' to save to open_eyes
    if key == ord('o') and 'eye_rois_to_save' in locals() and len(eye_rois_to_save) > 0:
        for eye in eye_rois_to_save:
            cv2.imwrite(os.path.join(OPEN_EYES_PATH, f'{uuid.uuid1()}.jpg'), eye)
        print(f"Saved to open_eyes - Total: {len(os.listdir(OPEN_EYES_PATH))}")

    # Press 'c' to save to closed_eyes
    if key == ord('c') and 'eye_rois_to_save' in locals() and len(eye_rois_to_save) > 0:
        for eye in eye_rois_to_save:
            cv2.imwrite(os.path.join(CLOSED_EYES_PATH, f'{uuid.uuid1()}.jpg'), eye)
        print(f"Saved to closed_eyes - Total: {len(os.listdir(CLOSED_EYES_PATH))}")

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()