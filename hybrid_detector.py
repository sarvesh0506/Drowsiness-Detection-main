import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np

# --- 1. DEFINITIONS AND PARAMETERS ---

def eye_aspect_ratio(eye):
    # Compute vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for EAR below which the eye is considered closed
EYE_AR_THRESH = 0.25

# Number of consecutive frames for the alert (7 seconds at ~30 FPS)
EYE_AR_CONSEC_FRAMES = 210 

# Initialize the frame counter
COUNTER = 0

# --- 2. INITIALIZE DLIB MODELS ---

print("[INFO] Loading dlib models...")
# This is dlib's built-in face detector
detector = dlib.get_frontal_face_detector()
# This is dlib's landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indexes for left and right eye landmarks
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# --- 3. START VIDEO STREAM ---

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using dlib's detector
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        # Get the facial landmarks
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Extract eye coordinates and calculate EAR
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw the eye contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Check for drowsiness
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        # Display the EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Drowsiness Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()