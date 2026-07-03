import cv2
import numpy as np
from scipy.spatial import distance as dist

# MediaPipe Face Mesh Indices for Eyes & Mouth
# Left Eye (Anatomical Left / Camera Right)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Right Eye (Anatomical Right / Camera Left)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 380, 373]
# Inner Lip / Mouth Perimeter
MOUTH_INDICES = [78, 81, 311, 308, 317, 88]

# 3D Model Points of a generic face for Head Pose Estimation (solvePnP)
# Coordinate system: Nose tip is the origin (0,0,0)
# Points: Nose tip, Chin, Left eye left corner, Right eye right corner, Left mouth corner, Right mouth corner
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 1)
    (0.0, -330.0, -65.0),        # Chin (index 152)
    (-225.0, 170.0, -135.0),     # Left Eye outer corner (index 33)
    (225.0, 170.0, -135.0),      # Right Eye outer corner (index 263)
    (-150.0, -150.0, -125.0),    # Left Mouth corner (index 61)
    (150.0, -150.0, -125.0)      # Right Mouth corner (index 291)
], dtype=np.float32)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR).
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    Points layout:
        p2     p3
     p1           p4
        p6     p5
    """
    if len(eye_points) < 6:
        return 0.0
    
    # Vertical distances
    a = calculate_distance(eye_points[1], eye_points[5])
    b = calculate_distance(eye_points[2], eye_points[4])
    # Horizontal distance
    c = calculate_distance(eye_points[0], eye_points[3])
    
    # Avoid division by zero
    if c < 1e-6:
        return 0.0
        
    ear = (a + b) / (2.0 * c)
    return ear

def calculate_mar(mouth_points):
    """
    Calculate Mouth Aspect Ratio (MAR) for yawn detection.
    Formula: MAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    Points layout:
        p2     p3
     p1           p4
        p6     p5
    """
    if len(mouth_points) < 6:
        return 0.0
        
    # Vertical distances
    a = calculate_distance(mouth_points[1], mouth_points[5])
    b = calculate_distance(mouth_points[2], mouth_points[4])
    # Horizontal distance
    c = calculate_distance(mouth_points[0], mouth_points[3])
    
    if c < 1e-6:
        return 0.0
        
    mar = (a + b) / (2.0 * c)
    return mar

def estimate_head_pose(landmarks, img_w, img_h):
    """
    Estimate Head Pose (Pitch, Yaw, Roll) in degrees using cv2.solvePnP.
    landmarks: A list of 468/478 landmarks from MediaPipe Face Mesh.
    Returns: pitch, yaw, roll, and projections points for drawing the direction vector.
    """
    # Map landmark indexes to the 3D model coordinates
    # MediaPipe indexes:
    # 1: Nose tip
    # 152: Chin
    # 33: Anatomical left eye outer corner (camera right)
    # 263: Anatomical right eye outer corner (camera left)
    # 61: Anatomical left mouth corner
    # 291: Anatomical right mouth corner
    idx_map = [1, 152, 33, 263, 61, 291]
    
    face_2d = []
    for idx in idx_map:
        lm = landmarks[idx]
        face_2d.append([lm.x * img_w, lm.y * img_h])
        
    face_2d = np.array(face_2d, dtype=np.float32)
    
    # Camera intrinsic matrix estimation
    # Focal length approximate
    focal_length = img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1), dtype=np.float32) # Assume zero distortion
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return 0.0, 0.0, 0.0, None
        
    # Get rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Compute Euler angles (Pitch, Yaw, Roll)
    # pitch: nodding up/down
    # yaw: shaking head left/right
    # roll: tilting head side to side
    
    # Projection angle decomposition
    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0
        
    # Convert to degrees
    pitch = np.degrees(x)
    yaw = np.degrees(y)
    roll = np.degrees(z)
    
    # Adjust angles based on calibration frame
    # (Typically raw Pitch needs adjustment depending on head angle mapping)
    pitch = -pitch
    
    # Project 3D nose axis point into 2D to draw visual vector
    # Project a point 300 units out along the Z axis from the nose
    axis_3d = np.array([(0.0, 0.0, 300.0)], dtype=np.float32)
    nose_2d_proj, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_matrix, dist_coeffs)
    
    nose_tip_2d = (int(face_2d[0][0]), int(face_2d[0][1]))
    nose_proj_2d = (int(nose_2d_proj[0][0][0]), int(nose_2d_proj[0][0][1]))
    
    return pitch, yaw, roll, (nose_tip_2d, nose_proj_2d)

def preprocess_eye_image(eye_image, target_size=(64, 64)):
    """Preprocess eye image crop for CNN classification."""
    try:
        # Resize to standard size
        resized = cv2.resize(eye_image, target_size)
        # Convert to RGB (if BGR)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        # Expand dimensions for batch size [1, H, W, C]
        expanded = np.expand_dims(normalized, axis=0)
        return expanded
    except Exception as e:
        # Fallback return empty batch
        return np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)
