import os
import threading
import logging
import time

logger = logging.getLogger(__name__)

# Try to import pygame for audio playback. Handle import error if not installed.
PYGAME_AVAILABLE = False
try:
    # Set environment variable to suppress pygame welcome banner
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    logger.warning("pygame library not found. Audio alarms will fall back to console logging/system beep.")

class AlarmManager:
    """Thread-safe manager for triggering and controlling driver alarms."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one AlarmManager controls audio output."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AlarmManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, sound_path=None):
        if self._initialized:
            return
            
        self.sound_path = sound_path
        self.is_playing = False
        self.is_muted = False
        self.alarm_thread = None
        self.stop_event = threading.Event()
        
        # Initialize Pygame Mixer if available
        self.mixer_initialized = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                self.mixer_initialized = True
                logger.info("Pygame mixer initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Pygame mixer (likely no audio device): {e}")
                
        self._initialized = True

    def _play_loop(self):
        """Worker function for playing the alarm sound in a background loop."""
        logger.info("Alarm thread started.")
        
        # Determine if we can play audio
        has_audio_device = self.mixer_initialized and self.sound_path and os.path.exists(self.sound_path)
        
        if has_audio_device and not self.is_muted:
            try:
                pygame.mixer.music.load(self.sound_path)
                pygame.mixer.music.play(-1) # Loop indefinitely
                
                # Wait until stopped
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                    
                pygame.mixer.music.stop()
            except Exception as e:
                logger.error(f"Error during audio playback loop: {e}")
                has_audio_device = False # Fallback to beep loop
                
        if not has_audio_device and not self.is_muted:
            # Fallback beep/console logging loop
            while not self.stop_event.is_set():
                logger.warning("!!! DROWSINESS ALARM ACTIVE - DRIVER SLEEPING !!!")
                # Cross-platform system alert beep (Windows specific or ASCII bell)
                try:
                    if os.name == 'nt':
                        import winsound
                        winsound.Beep(1000, 500) # 1000 Hz for 500ms
                    else:
                        print('\a', end='', flush=True) # Ascii bell
                        time.sleep(0.5)
                except Exception as e:
                    logger.debug(f"Beep fallback error: {e}")
                    time.sleep(0.5)
                    
        self.is_playing = False
        logger.info("Alarm thread finished.")

    def trigger(self):
        """Trigger the alarm in a non-blocking thread."""
        with self._lock:
            if self.is_playing:
                return # Already playing
                
            if self.is_muted:
                logger.info("Alarm triggered but is currently MUTED.")
                return
                
            logger.info("Triggering alarm...")
            self.is_playing = True
            self.stop_event.clear()
            self.alarm_thread = threading.Thread(target=self._play_loop, name="AlarmThread")
            self.alarm_thread.daemon = True
            self.alarm_thread.start()

    def stop(self):
        """Stop the running alarm thread."""
        with self._lock:
            if not self.is_playing:
                return
                
            logger.info("Stopping alarm...")
            self.stop_event.set()
            if self.alarm_thread:
                self.alarm_thread.join(timeout=1.0)
            self.is_playing = False

    def set_mute(self, is_muted):
        """Mute/unmute the alarm."""
        with self._lock:
            logger.info(f"Setting alarm mute state to: {is_muted}")
            self.is_muted = is_muted
            if is_muted and self.is_playing:
                # Stop if currently playing
                self.stop_event.set()
                self.is_playing = False
