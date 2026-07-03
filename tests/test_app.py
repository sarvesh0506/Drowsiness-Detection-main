import os
import json
import base64
import unittest
import numpy as np
import cv2
from app import app, camera, alarm
from utils import calculate_distance, calculate_ear, calculate_mar

class DrowsinessDetectionTestCase(unittest.TestCase):
    
    def setUp(self):
        # Bind testing configurations
        app.config['TESTING'] = True
        app.config['DEBUG'] = True
        self.client = app.test_client()
        
        # Mute alarm for tests to prevent system speaker noises in build systems
        alarm.set_mute(True)
        
    def tearDown(self):
        # Stop background timers/alarms
        alarm.stop()
        camera.stop()

    # --- Unit Tests for Utility Algebra Math ---
    
    def test_calculate_distance(self):
        """Test Euclidean distance calculation helper."""
        p1 = [0, 0]
        p2 = [3, 4]
        self.assertEqual(calculate_distance(p1, p2), 5.0)

    def test_calculate_ear_normal(self):
        """Test Eye Aspect Ratio calculation on structured coordinates."""
        # 6 Points representing open eye shape:
        # P1 (0, 5), P2 (5, 10), P3 (15, 10), P4 (20, 5), P5 (15, 0), P6 (5, 0)
        eye_pts = np.array([
            [0, 5],   # P1
            [5, 10],  # P2
            [15, 10], # P3
            [20, 5],  # P4
            [15, 0],  # P5
            [5, 0]    # P6
        ])
        
        # expected vertical distance: (||p2-p6|| + ||p3-p5||) = (10 + 10) = 20
        # expected horizontal distance: 2 * ||p1-p4|| = 2 * 20 = 40
        # ear = 20 / 40 = 0.5
        ear = calculate_ear(eye_pts)
        self.assertAlmostEqual(ear, 0.5)

    def test_calculate_ear_zero_division(self):
        """Verify EAR handles invalid/collapsed coordinates without DivisionByZero crash."""
        # Flat coordinates (P1 and P4 identical)
        eye_pts = np.array([
            [0, 0], [0, 5], [0, 5], [0, 0], [0, 0], [0, 0]
        ])
        ear = calculate_ear(eye_pts)
        self.assertEqual(ear, 0.0)

    def test_calculate_mar_normal(self):
        """Test Mouth Aspect Ratio calculation."""
        mouth_pts = np.array([
            [0, 5],   # P1
            [5, 12],  # P2
            [15, 12], # P3
            [20, 5],  # P4
            [15, -2], # P5
            [5, -2]   # P6
        ])
        # vertical = (14 + 14) = 28
        # horizontal = 2 * 20 = 40
        # mar = 28 / 40 = 0.70
        mar = calculate_mar(mouth_pts)
        self.assertAlmostEqual(mar, 0.70)

    # --- Integration Tests for Flask REST Endpoints ---

    def test_dashboard_route(self):
        """Verify index.html page renders status 200 and loads core template containers."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        html = response.data.decode('utf-8')
        self.assertIn('AI DRIVER GUARDIAN', html)
        self.assertIn('LIVE GUARDIAN STREAM', html)
        self.assertIn('TELEMETRY GAUGES', html)
        self.assertIn('biometricChart', html)

    def test_api_status_structure(self):
        """Check status API returns expected keys and correct data types."""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data.decode('utf-8'))
        
        expected_keys = [
            "ear", "mar", "pitch", "yaw", "roll", "fatigue_score",
            "blink_count", "blink_rate", "yawn_count", "status",
            "alarm_active", "alarm_muted", "model_loaded", "fps"
        ]
        for key in expected_keys:
            self.assertIn(key, data)
            
        self.assertIsInstance(data['ear'], float)
        self.assertIsInstance(data['fatigue_score'], (int, float))
        self.assertIsInstance(data['blink_count'], int)
        self.assertIsInstance(data['status'], str)

    def test_api_settings_get_post(self):
        """Assert configuration GET returns default values and POST updates settings."""
        # 1. Test GET Settings
        response_get = self.client.get('/api/settings')
        self.assertEqual(response_get.status_code, 200)
        data_get = json.loads(response_get.data.decode('utf-8'))
        
        self.assertIn('ear_threshold', data_get)
        self.assertIn('mar_threshold', data_get)
        
        # 2. Test POST Settings overrides
        payload = {
            "ear_threshold": 0.22,
            "mar_threshold": 0.72,
            "closed_frames": 12,
            "yawn_frames": 22,
            "alarm_muted": True
        }
        response_post = self.client.post(
            '/api/settings',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response_post.status_code, 200)
        
        data_post = json.loads(response_post.data.decode('utf-8'))
        self.assertTrue(data_post['success'])
        
        # Verify settings modified in camera instance
        self.assertAlmostEqual(camera.ear_threshold, 0.22)
        self.assertAlmostEqual(camera.mar_threshold, 0.72)
        self.assertEqual(camera.closed_frames_threshold, 12)
        self.assertEqual(camera.yawn_frames_threshold, 22)
        self.assertTrue(alarm.is_muted)

    def test_api_alarm_endpoints(self):
        """Test API endpoints for manually triggering and stopping the audio alarms."""
        # Trigger
        response_trigger = self.client.post('/api/alarm/trigger')
        self.assertEqual(response_trigger.status_code, 200)
        data_trigger = json.loads(response_trigger.data.decode('utf-8'))
        self.assertTrue(data_trigger['success'])
        
        # Stop
        response_stop = self.client.post('/api/alarm/stop')
        self.assertEqual(response_stop.status_code, 200)
        data_stop = json.loads(response_stop.data.decode('utf-8'))
        self.assertTrue(data_stop['success'])

    def test_api_process_frame_invalid(self):
        """Test API frame processing handles empty/invalid payloads gracefully."""
        # Empty payload
        response = self.client.post(
            '/api/process_frame',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
    def test_api_process_frame_valid_shape(self):
        """Validate process_frame_api successfully ingests and parses a blank base64 frame."""
        # Generate a small 100x100 solid color OpenCV image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', img)
        base64_str = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        
        payload = {
            "image": f"data:image/jpeg;base64,{base64_str}"
        }
        
        response = self.client.post(
            '/api/process_frame',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data.decode('utf-8'))
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'No Face Detected') # Blank frame has no face landmarks
        self.assertIn('processed_image', data)
        self.assertTrue(data['processed_image'].startswith('data:image/jpeg;base64,'))

if __name__ == '__main__':
    unittest.main()
