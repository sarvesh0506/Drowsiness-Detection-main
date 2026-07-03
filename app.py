import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, Response, jsonify, request, url_safe_get_jwt_header
from config import get_config
from camera import VideoCamera
from alarm import AlarmManager

# Configure root logger
config = get_config()
if not os.path.exists(config.LOG_DIR):
    os.makedirs(config.LOG_DIR)

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger('app')

# Setup rotating file handler for production grade logging
file_handler = RotatingFileHandler(config.LOG_FILE_PATH, maxBytes=1024*1024*5, backupCount=5)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s:%(lineno)d] - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
logging.getLogger().addHandler(file_handler)

app = Flask(__name__)
app.config.from_object(config)

# Initialize singletons
camera = VideoCamera()
alarm = AlarmManager(config.ALARM_SOUND_PATH)

# Create static asset directories if missing
os.makedirs(config.STATIC_AUDIO_DIR, exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'js'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'uploads'), exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

# Generate dummy files if they don't exist to ensure out-of-the-box runnability
# Create a dummy alert.mp3 (simple 1-second silent audio file or mock audio)
# (In production, the real alert.mp3 will be loaded. Pygame handles lack of file gracefully)
dummy_audio_path = config.ALARM_SOUND_PATH
if not os.path.exists(dummy_audio_path):
    try:
        # Create an empty file; pygame will log warning, alarm will fallback to system beep
        with open(dummy_audio_path, 'wb') as f:
            f.write(b'')
        logger.info(f"Created placeholder alert audio file at {dummy_audio_path}")
    except Exception as e:
        logger.error(f"Failed to create dummy audio file: {e}")

@app.route('/')
def index():
    """Render the dashboard UI."""
    return render_template('index.html')

def gen_frames(cam):
    """Generator for streaming camera frames via MJPEG."""
    # Start camera capture thread if not running
    cam.start()
    
    try:
        while True:
            frame_bytes = cam.get_frame()
            if frame_bytes is None:
                # Sleep briefly if camera has no frame yet
                time.sleep(0.03)
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        logger.info("MJPEG client disconnected.")
    except Exception as e:
        logger.error(f"Error in gen_frames: {e}")

@app.route('/video_feed')
def video_feed():
    """Multipart stream of processed local camera feed."""
    if camera.config.CAMERA_SOURCE is None:
        return jsonify({"error": "Local camera source is disabled/None in settings."}), 400
        
    return Response(gen_frames(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/process_frame', methods=['POST'])
def process_frame_api():
    """
    POST base64 image frame from Web UI.
    Used for cloud deployment when the server lacks a camera.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No image payload found"}), 400
        
    base64_image = data['image']
    
    # Input sanitization and size check (max 10MB)
    if len(base64_image) > 10 * 1024 * 1024:
        return jsonify({"success": False, "error": "Payload size exceeded limit (10MB)"}), 413
        
    response = camera.process_base64_frame(base64_image)
    return jsonify(response)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Retrieve current metrics and model loading state."""
    # Compute active blink frequency
    import time
    now = time.time()
    camera.blink_time_window = [t for t in camera.blink_time_window if now - t < 60]
    
    status_data = {
        "ear": camera.ear,
        "mar": camera.mar,
        "pitch": camera.pitch,
        "yaw": camera.yaw,
        "roll": camera.roll,
        "fatigue_score": camera.fatigue_score,
        "blink_count": camera.blink_count,
        "blink_rate": len(camera.blink_time_window),
        "yawn_count": camera.yawn_count,
        "status": camera.status,
        "alarm_active": alarm.is_playing,
        "alarm_muted": alarm.is_muted,
        "model_loaded": camera.model_loaded,
        "tf_load_in_progress": camera.tf_load_started,
        "fps": camera.fps,
        "camera_active": camera.is_running
    }
    return jsonify(status_data)

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """GET current configurations or POST overrides."""
    if request.method == 'POST':
        data = request.get_json() or {}
        
        # Parse and sanitize thresholds
        try:
            if 'ear_threshold' in data:
                camera.ear_threshold = max(0.1, min(0.4, float(data['ear_threshold'])))
            if 'mar_threshold' in data:
                camera.mar_threshold = max(0.2, min(1.0, float(data['mar_threshold'])))
            if 'closed_frames' in data:
                camera.closed_frames_threshold = max(5, min(100, int(data['closed_frames'])))
            if 'yawn_frames' in data:
                camera.yawn_frames_threshold = max(5, min(100, int(data['yawn_frames'])))
            if 'alarm_muted' in data:
                alarm.set_mute(bool(data['alarm_muted']))
                
            logger.info("Application thresholds updated via settings API.")
            return jsonify({"success": True, "message": "Settings updated successfully."})
        except (ValueError, TypeError) as e:
            return jsonify({"success": False, "error": f"Invalid settings input: {e}"}), 400
            
    # GET settings
    return jsonify({
        "ear_threshold": camera.ear_threshold,
        "mar_threshold": camera.mar_threshold,
        "closed_frames_threshold": camera.closed_frames_threshold,
        "yawn_frames_threshold": camera.yawn_frames_threshold,
        "alarm_muted": alarm.is_muted,
        "camera_source": str(config.CAMERA_SOURCE)
    })

@app.route('/api/alarm/trigger', methods=['POST'])
def trigger_alarm():
    """Manual alarm trigger endpoint."""
    alarm.trigger()
    return jsonify({"success": True, "message": "Alarm triggered."})

@app.route('/api/alarm/stop', methods=['POST'])
def stop_alarm():
    """Manual alarm stop endpoint."""
    alarm.stop()
    return jsonify({"success": True, "message": "Alarm stopped."})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Retrieve the recent logs from application logger."""
    lines_count = request.args.get('lines', 50, type=int)
    log_content = []
    
    if os.path.exists(config.LOG_FILE_PATH):
        try:
            with open(config.LOG_FILE_PATH, 'r') as f:
                log_content = f.readlines()[-lines_count:]
        except Exception as e:
            return jsonify({"error": f"Unable to read log file: {e}"}), 500
            
    return Response("".join(log_content), mimetype='text/plain')

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle oversized file uploads/payloads."""
    return jsonify({"success": False, "error": "File/payload size exceeds security limit (16MB)"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle unexpected server errors gracefully."""
    logger.error(f"Internal Server Error: {error}")
    return jsonify({"success": False, "error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # When run directly, bind to all interfaces (ports 5000)
    # Start local camera if a valid source is provided in config
    if config.CAMERA_SOURCE is not None:
        camera.start()
        
    try:
        app.run(host='0.0.0.0', port=5000, debug=config.DEBUG, use_reloader=False)
    finally:
        # Graceful shutdown of webcam
        camera.stop()
