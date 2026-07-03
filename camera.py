import os
import cv2
import numpy as np
import threading
import time
import base64
import logging
from config import get_config
from utils import calculate_ear, calculate_mar, estimate_head_pose, preprocess_eye_image, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES
from alarm import AlarmManager

# MediaPipe Face Mesh
try:
    import mediapipe as mp
    from mediapipe.solutions import face_mesh as mp_face_mesh
except ImportError:
    mp = None
    mp_face_mesh = None

logger = logging.getLogger(__name__)

class VideoCamera:
    """Threaded camera processing engine executing MediaPipe Face Mesh and CNN classifiers."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for VideoCamera to avoid multiple camera captures."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VideoCamera, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config = get_config()
        self.alarm_manager = AlarmManager(self.config.ALARM_SOUND_PATH)
        
        # Camera capturing parameters
        self.video = None
        self.is_running = False
        self.frame = None
        self.processed_frame = None
        self.capture_thread = None
        
        # Current AI metrics
        self.ear = 0.0
        self.mar = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.fps = 0.0
        
        # State Machine Counters
        self.consec_closed_frames = 0
        self.consec_yawn_frames = 0
        self.consec_pose_frames = 0
        
        self.blink_count = 0
        self.yawn_count = 0
        self.fatigue_score = 0.0
        self.status = "Alert"
        
        # Blink tracking helpers
        self.eye_was_closed = False
        self.blink_time_window = [] # Timestamps of recent blinks to estimate frequency
        self.yawn_was_active = False
        
        # Threshold overrides (can be updated dynamically via settings API)
        self.ear_threshold = self.config.DEFAULT_EAR_THRESHOLD
        self.mar_threshold = self.config.DEFAULT_MAR_THRESHOLD
        self.closed_frames_threshold = self.config.DEFAULT_CLOSED_FRAMES
        self.yawn_frames_threshold = self.config.DEFAULT_YAWN_FRAMES
        
        # MediaPipe initialization
        self.mp_face_mesh = None
        self.face_mesh = None
        if mp and mp_face_mesh:
            try:
                self.mp_face_mesh = mp_face_mesh
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe Face Mesh initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe Face Mesh: {e}")
                
        # TensorFlow CNN Classifier (Lazy loaded)
        self.model = None
        self.model_loaded = False
        self.tf_load_started = False
        
        # Trigger lazy load in background thread to speed up initialization
        self.trigger_model_load()
        
        self._initialized = True

    def trigger_model_load(self):
        """Starts a background thread to load the Keras eye classifier model."""
        if not self.tf_load_started and not self.model_loaded:
            self.tf_load_started = True
            t = threading.Thread(target=self._load_model_worker, name="TensorFlowLoader")
            t.daemon = True
            t.start()

    def _load_model_worker(self):
        """Worker thread to load TensorFlow model."""
        model_path = self.config.EYE_CLASSIFIER_PATH
        if os.path.exists(model_path):
            try:
                logger.info("Loading TensorFlow Eye Classifier in background...")
                # Import TensorFlow lazily inside thread
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                logger.info(f"TensorFlow eye classifier model loaded successfully from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load eye classifier model: {e}. Falling back to EAR logic.")
        else:
            logger.warning(f"Model path {model_path} does not exist. Running in geometric-only EAR mode.")
            
        self.tf_load_started = False

    def start(self):
        """Start the threaded camera capture."""
        with self._lock:
            if self.is_running:
                return
                
            src = self.config.CAMERA_SOURCE
            if src is None:
                logger.warning("Camera source configured as None. Server capturing disabled.")
                return
                
            self.video = cv2.VideoCapture(src)
            if not self.video.isOpened():
                logger.error(f"Could not open camera source: {src}")
                return
                
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_worker, name="CameraCaptureThread")
            self.capture_thread.daemon = True
            self.capture_thread.start()
            logger.info("Threaded camera capture started.")

    def stop(self):
        """Stop camera capture and release resources."""
        with self._lock:
            self.is_running = False
            self.alarm_manager.stop()
            if self.capture_thread:
                self.capture_thread.join(timeout=1.0)
            if self.video:
                self.video.release()
                self.video = None
            logger.info("Threaded camera capture stopped.")

    def _capture_worker(self):
        """Reads frames in background as fast as the camera allows."""
        prev_time = time.time()
        while self.is_running:
            success, raw_frame = self.video.read()
            if not success:
                logger.warning("Failed to read frame from camera. Reconnecting...")
                time.sleep(0.5)
                continue
                
            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            self.fps = 1.0 / time_diff if time_diff > 0 else 30.0
            prev_time = curr_time
            
            # Update frame buffer
            self.frame = raw_frame.copy()
            
            # Process current frame
            self.processed_frame = self.process_frame(self.frame)
            
            # Limit thread speed
            time.sleep(0.01)

    def get_frame(self):
        """Return the processed frame encoded in jpeg."""
        if self.processed_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', self.processed_frame)
            if ret:
                return jpeg.tobytes()
        return None

    def process_frame(self, frame):
        """
        Process frame through MediaPipe and updates AI metrics.
        Returns the annotated frame.
        """
        if frame is None:
            return None
            
        img_h, img_w, _ = frame.shape
        annotated_frame = frame.copy()
        
        # If MediaPipe is not loaded/fails, return raw frame
        if self.face_mesh is None:
            return annotated_frame
            
        # MediaPipe requires RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. EAR (Eye Aspect Ratio) Calculation
            left_eye_pts = np.array([[landmarks[idx].x * img_w, landmarks[idx].y * img_h] for idx in LEFT_EYE_INDICES])
            right_eye_pts = np.array([[landmarks[idx].x * img_w, landmarks[idx].y * img_h] for idx in RIGHT_EYE_INDICES])
            
            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            self.ear = (left_ear + right_ear) / 2.0
            
            # 2. CNN Eye State prediction (if model loaded)
            cnn_closed_conf = 0.0
            if self.model_loaded:
                # Crop left and right eyes
                left_eye_crop = self.crop_eye(frame, left_eye_pts, img_w, img_h)
                right_eye_crop = self.crop_eye(frame, right_eye_pts, img_w, img_h)
                
                if left_eye_crop is not None and right_eye_crop is not None:
                    left_batch = preprocess_eye_image(left_eye_crop)
                    right_batch = preprocess_eye_image(right_eye_crop)
                    
                    # Predict state (0 for open, 1 for closed)
                    left_pred = self.model.predict(left_batch, verbose=0)[0][0]
                    right_pred = self.model.predict(right_batch, verbose=0)[0][0]
                    
                    cnn_closed_conf = (left_pred + right_pred) / 2.0
            
            # 3. Yawn (Mouth Aspect Ratio) Detection
            mouth_pts = np.array([[landmarks[idx].x * img_w, landmarks[idx].y * img_h] for idx in MOUTH_INDICES])
            self.mar = calculate_mar(mouth_pts)
            
            # 4. Head Pose Estimation
            self.pitch, self.yaw, self.roll, pose_vectors = estimate_head_pose(landmarks, img_w, img_h)
            
            # Determine if eyes are closed (Combine EAR & CNN predictions)
            # If CNN loaded, use weighted fusion: 60% CNN, 40% EAR geometric threshold
            is_eye_closed = False
            if self.model_loaded:
                # Normalize EAR closed score: maps ear values below threshold to 1.0, above to 0.0
                ear_closed_val = max(0.0, min(1.0, (self.ear_threshold - self.ear) / self.ear_threshold))
                fused_closed_score = 0.6 * cnn_closed_conf + 0.4 * ear_closed_val
                is_eye_closed = fused_closed_score > 0.5
            else:
                is_eye_closed = self.ear < self.ear_threshold
                
            # --- Alert State Machine Logic ---
            
            # Drowsiness tracking
            if is_eye_closed:
                self.consec_closed_frames += 1
                self.eye_was_closed = True
            else:
                # Check if it was a blink (closure duration was short)
                if self.eye_was_closed and self.consec_closed_frames < self.closed_frames_threshold:
                    self.blink_count += 1
                    # Record blink time to update frequency
                    self.blink_time_window.append(time.time())
                
                self.consec_closed_frames = 0
                self.eye_was_closed = False
                
            # Yawn tracking
            if self.mar > self.mar_threshold:
                self.consec_yawn_frames += 1
                if not self.yawn_was_active and self.consec_yawn_frames >= self.yawn_frames_threshold:
                    self.yawn_count += 1
                    self.yawn_was_active = True
                    self.fatigue_score = min(100.0, self.fatigue_score + 15.0) # Boost fatigue for yawn
            else:
                self.consec_yawn_frames = 0
                self.yawn_was_active = False
                
            # Distraction (Pose) tracking
            is_head_distracted = (
                self.pitch > self.config.DEFAULT_POSE_PITCH_UP or
                self.pitch < self.config.DEFAULT_POSE_PITCH_DOWN or
                self.yaw > self.config.DEFAULT_POSE_YAW_RIGHT or
                self.yaw < self.config.DEFAULT_POSE_YAW_LEFT
            )
            
            if is_head_distracted:
                self.consec_pose_frames += 1
            else:
                self.consec_pose_frames = 0
                
            # Update Fatigue Score and Alarm triggering
            if self.consec_closed_frames >= self.closed_frames_threshold:
                # Rapidly increase fatigue for prolonged closure
                self.fatigue_score = min(100.0, self.fatigue_score + 4.0)
                self.status = "Drowsy"
                self.alarm_manager.trigger()
            elif self.consec_pose_frames >= self.config.DEFAULT_POSE_CONSEC_FRAMES:
                # Increment fatigue for distraction
                self.fatigue_score = min(100.0, self.fatigue_score + 0.8)
                self.status = "Distracted"
                self.alarm_manager.stop() # Head turned is distraction, not immediate micro-sleep alarm
            else:
                # Recovery when driver is alert
                self.fatigue_score = max(0.0, self.fatigue_score - 0.2)
                self.alarm_manager.stop()
                
                # Update status classification based on fatigue score levels
                if self.fatigue_score > 70.0:
                    self.status = "Drowsy"
                    self.alarm_manager.trigger()
                elif self.fatigue_score > 40.0:
                    self.status = "Warning"
                else:
                    self.status = "Alert"
                    
            # Draw facial landmarks
            self.draw_overlays(annotated_frame, left_eye_pts, right_eye_pts, mouth_pts, pose_vectors, is_eye_closed)
            
        else:
            # No face detected
            self.consec_closed_frames = 0
            self.consec_yawn_frames = 0
            self.consec_pose_frames = 0
            self.status = "No Face Detected"
            self.alarm_manager.stop()
            
        # Draw Dashboard panel overlay directly onto frame (for server-side video feed)
        self.draw_dashboard_overlay(annotated_frame, face_detected)
        
        return annotated_frame

    def crop_eye(self, frame, eye_pts, img_w, img_h):
        """Helper to crop the eye region from frame with bounding box padding."""
        try:
            xs = eye_pts[:, 0]
            ys = eye_pts[:, 1]
            min_x, max_x = int(np.min(xs)), int(np.max(xs))
            min_y, max_y = int(np.min(ys)), int(np.max(ys))
            
            # Add padding
            pad_w = int((max_x - min_x) * 0.2)
            pad_h = int((max_y - min_y) * 0.4)
            
            x1 = max(0, min_x - pad_w)
            y1 = max(0, min_y - pad_h)
            x2 = min(img_w, max_x + pad_w)
            y2 = min(img_h, max_y + pad_h)
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                return frame[y1:y2, x1:x2]
        except Exception as e:
            logger.debug(f"Eye crop error: {e}")
        return None

    def draw_overlays(self, frame, left_eye, right_eye, mouth, pose_vectors, is_eye_closed):
        """Draw outlines on facial features and project pose vectors."""
        # Color palettes (B, G, R)
        alert_color = (0, 230, 118) # Neon Emerald
        drowsy_color = (0, 0, 255)  # Bright Red
        
        eye_color = drowsy_color if is_eye_closed else alert_color
        yawn_color = drowsy_color if self.mar > self.mar_threshold else alert_color
        
        # Draw eyes polygons
        cv2.polylines(frame, [left_eye.astype(np.int32)], True, eye_color, 1)
        cv2.polylines(frame, [right_eye.astype(np.int32)], True, eye_color, 1)
        
        # Draw mouth
        cv2.polylines(frame, [mouth.astype(np.int32)], True, yawn_color, 1)
        
        # Draw eye status text overlay
        eye_label = "CLOSED" if is_eye_closed else "OPEN"
        cv2.putText(frame, f"EYES: {eye_label}", (int(left_eye[0][0] - 10), int(left_eye[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, eye_color, 1)
                    
        # Draw Head Pose Direction vector
        if pose_vectors:
            nose_tip, nose_proj = pose_vectors
            # Draw line projecting outwards
            cv2.line(frame, nose_tip, nose_proj, (255, 235, 59), 2) # Yellow vector line
            cv2.circle(frame, nose_proj, 4, (0, 145, 234), -1)      # Blue pointer tip

    def draw_dashboard_overlay(self, frame, face_detected):
        """Draw aesthetic UI dashboard details on frame."""
        # Glassmorphism dark background box
        # cv2.rectangle (BGR: bottom-left, top-right)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (220, 160), (32, 22, 16), -1) # Semi-dark box
        cv2.rectangle(frame, (10, 10), (220, 160), (80, 80, 80), 1)  # Border
        
        # Panel Text info
        # BGR colors
        stat_colors = {
            "Alert": (0, 230, 118),      # Emerald
            "Warning": (0, 145, 255),    # Orange
            "Distracted": (0, 215, 255), # Yellow-gold
            "Drowsy": (0, 0, 255),       # Red
            "No Face Detected": (150, 150, 150)
        }
        
        status_color = stat_colors.get(self.status, (255, 255, 255))
        
        cv2.putText(frame, "AI DRIVER GUARDIAN", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (20, 36), (210, 36), (60, 60, 60), 1)
        
        cv2.putText(frame, f"STATUS: {self.status}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"EAR: {self.ear:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"MAR: {self.mar:.3f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FATIGUE: {self.fatigue_score:.1f}%", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)
        
        pose_str = f"P:{self.pitch:.1f} Y:{self.yaw:.1f}"
        cv2.putText(frame, f"POSE: {pose_str}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 229, 255), 1, cv2.LINE_AA)

    def process_base64_frame(self, base64_str):
        """
        Process an image string (base64) received from client UI.
        Used when application is running in the cloud.
        """
        try:
            # Decode base64
            img_data = base64.b64decode(base64_str.split(',')[1] if ',' in base64_str else base64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process metrics on this frame
            processed = self.process_frame(img)
            
            # Encode response image back to base64
            ret, jpeg = cv2.imencode('.jpg', processed)
            encoded_img = ""
            if ret:
                encoded_img = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                
            # Clean up blink window to compute blink frequency (blinks in last 60 seconds)
            now = time.time()
            self.blink_time_window = [t for t in self.blink_time_window if now - t < 60]
            blink_rate_per_min = len(self.blink_time_window)
            
            return {
                "success": True,
                "status": self.status,
                "ear": float(self.ear),
                "mar": float(self.mar),
                "pitch": float(self.pitch),
                "yaw": float(self.yaw),
                "roll": float(self.roll),
                "fatigue_score": float(self.fatigue_score),
                "blink_count": self.blink_count,
                "blink_rate": blink_rate_per_min,
                "yawn_count": self.yawn_count,
                "alarm_active": self.alarm_manager.is_playing,
                "processed_image": f"data:image/jpeg;base64,{encoded_img}"
            }
        except Exception as e:
            logger.error(f"Error in base64 frame processing: {e}")
            return {"success": False, "error": str(e)}
