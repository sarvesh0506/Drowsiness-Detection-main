from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import dlib
import numpy as np
import cv2
import base64
from scipy.spatial import distance as dist

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# --- EAR Calculation Function ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Drowsiness Detection Variables ---
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES_THRESHOLD = 45
COUNTER = 0

print("[INFO] Loading dlib models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# --- Serve Homepage ---
@app.route("/")
def home():
    return render_template("index.html")

# --- API Endpoint for Drowsiness Detection ---
@app.route("/detect", methods=["POST"])

def detect_drowsiness():
    global COUNTER

    print("Received image from frontend.")

    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    is_drowsy = False
    ear = 0
    face_detected = False

    rects = detector(gray, 0)

    if len(rects) > 0:
        face_detected = True
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        
        print(f"EAR: {ear:.3f}, COUNTER: {COUNTER}, Face Detected: {face_detected}, Is Drowsy: {is_drowsy}")


        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= CONSECUTIVE_FRAMES_THRESHOLD:
                is_drowsy = True
        else:
            COUNTER = 0
    else:
        COUNTER = 0

    return jsonify({"ear": ear, "is_drowsy": is_drowsy, "face_detected": face_detected})



if __name__ == "__main__":
    app.run(debug=True)
