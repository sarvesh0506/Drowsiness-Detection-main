from flask import Flask, render_template, request, jsonify
import cv2
import dlib
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import threading
import time

app = Flask(__name__)

# Thread safety
lock = threading.Lock()

# Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye aspect ratio function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds
EYE_AR_THRESH = 0.24
CLOSED_EYE_TIME = 7.0 # seconds


# Variables
last_closed_time = None
last_request_time = 0
MIN_INTERVAL = 0.1  # 100ms between requests
first_frame_saved = False

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    # This is the corrected line
    global last_closed_time, last_request_time, first_frame_saved

    # -------- Rate Limit -------- #
    current_time = time.time()
    if current_time - last_request_time < MIN_INTERVAL:
        return jsonify({'is_drowsy': False})
    last_request_time = current_time
    # ---------------------------- #

    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64 image
    img_data = data['image'].split(',')[1]
    img_bytes = BytesIO(base64.b64decode(img_data))
    img = np.array(Image.open(img_bytes))
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # =================== DEBUGGING BLOCK (Now fixed) ===================
    if not first_frame_saved:
        cv2.imwrite("debug_image.jpg", frame)
        print("\n>>> SUCCESS: Saved the first frame as debug_image.jpg. Please check the file. <<<\n")
        first_frame_saved = True
    # ===================================================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    print(f"Faces detected: {len(faces)}")
    is_drowsy = False

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        with lock:
            current_time = time.time()
            if ear < EYE_AR_THRESH:
                if last_closed_time is None:
                    last_closed_time = current_time
                elif (current_time - last_closed_time) >= CLOSED_EYE_TIME:
                    is_drowsy = True
            else:
                last_closed_time = None

        print(f"EAR: {ear:.3f}, Closed_for: {0 if last_closed_time is None else (current_time - last_closed_time):.2f}s")

    return jsonify({'is_drowsy': is_drowsy})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
