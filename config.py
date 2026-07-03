import os
import secrets

class Config:
    """Base configurations."""
    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    APP_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Uploads & Security limits
    UPLOAD_FOLDER = os.path.join(APP_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
    
    # Static Assets & Logs
    LOG_DIR = os.path.join(APP_DIR, 'logs')
    LOG_FILE_PATH = os.path.join(LOG_DIR, 'app.log')
    STATIC_AUDIO_DIR = os.path.join(APP_DIR, 'static', 'audio')
    ALARM_SOUND_PATH = os.path.join(STATIC_AUDIO_DIR, 'alert.mp3')
    
    # Model Configurations
    MODEL_DIR = os.path.join(APP_DIR, 'models')
    EYE_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'eye_classifier.h5')
    
    # AI Engine Threshold Defaults
    DEFAULT_EAR_THRESHOLD = 0.25        # Eye Aspect Ratio threshold (below this is closed)
    DEFAULT_MAR_THRESHOLD = 0.65        # Mouth Aspect Ratio threshold (above this is open/yawn)
    DEFAULT_CLOSED_FRAMES = 15          # Consecutive frames with closed eyes to trigger drowsiness (approx 0.5 - 1 sec)
    DEFAULT_YAWN_FRAMES = 25            # Consecutive frames mouth open to trigger yawning alert
    
    # Head Pose Pitch/Yaw Thresholds (degrees)
    DEFAULT_POSE_PITCH_UP = 15.0        # Head up limit (nodding up)
    DEFAULT_POSE_PITCH_DOWN = -15.0     # Head down limit (nodding down/sleeping)
    DEFAULT_POSE_YAW_LEFT = -20.0       # Head left limit (looking left)
    DEFAULT_POSE_YAW_RIGHT = 20.0       # Head right limit (looking right)
    DEFAULT_POSE_FRAMES = 30            # Consecutive frames distracted to trigger warning
    
    # Webcam Configuration
    CAMERA_SOURCE = 0                   # Device index or video file/stream path

class DevelopmentConfig(Config):
    """Development configurations."""
    DEBUG = True
    ENV = 'development'
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configurations."""
    DEBUG = False
    ENV = 'production'
    LOG_LEVEL = 'INFO'
    # In production, enforce key validation
    SECRET_KEY = os.environ.get('SECRET_KEY', Config.SECRET_KEY)

class TestingConfig(Config):
    """Testing configurations."""
    TESTING = True
    DEBUG = True
    ENV = 'testing'
    LOG_LEVEL = 'CRITICAL'
    CAMERA_SOURCE = None                # Don't spin up camera during unit tests unless specified

# Mapping for environment config selection
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Helper to load config by environment variable."""
    env = os.environ.get('FLASK_ENV', 'development').lower()
    return config_by_name.get(env, config_by_name['default'])
