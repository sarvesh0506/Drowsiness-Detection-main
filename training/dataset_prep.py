import os
import cv2
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure(base_dir='data'):
    """Create directories for model training datasets."""
    splits = ['train', 'val', 'test']
    classes = ['open_eyes', 'closed_eyes']
    
    for split in splits:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory verified: {path}")

def generate_synthetic_eye(is_closed=False, size=(64, 64)):
    """
    Generate a synthetic representation of an eye for testing the training scripts.
    Open Eye: Draws an outer eye contour, iris circle, and pupil.
    Closed Eye: Draws a straight/curved slit.
    """
    # Create gray background representing face skin tone
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 128
    
    # Randomly vary the skin tone slightly
    img = img + np.random.randint(-15, 15, size=img.shape, dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    center = (size[0] // 2, size[1] // 2)
    
    if not is_closed:
        # Draw open eye ellipse shape
        cv2.ellipse(img, center, (22, 12), 0, 0, 360, (230, 230, 230), -1) # sclera (white)
        cv2.ellipse(img, center, (22, 12), 0, 0, 360, (60, 60, 60), 1)     # eye border
        # Draw iris (colored part, e.g. blue/brown)
        cv2.circle(img, center, 8, (120, 80, 50), -1)
        # Draw pupil (black center)
        cv2.circle(img, center, 4, (10, 10, 10), -1)
        # Add eye highlight (light reflection)
        cv2.circle(img, (center[0] - 3, center[1] - 3), 1, (255, 255, 255), -1)
    else:
        # Draw closed eye slit (horizontal curve)
        # We draw a thick curved line representing eyelids closed
        axes = (20, 4)
        cv2.ellipse(img, (center[0], center[1] - 2), axes, 0, 10, 170, (40, 40, 40), 2)
        # Draw small eyelashes lines
        for lash_x in range(center[0] - 15, center[0] + 16, 6):
            cv2.line(img, (lash_x, center[1]), (lash_x - 2, center[1] + 4), (40, 40, 40), 1)

    # Apply slight gaussian blur to blend edges
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Add random pixel noise to simulate sensor noise
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = cv2.add(img, noise, dtype=cv2.CV_8U)
    
    return img

def build_mock_dataset(base_dir='data', samples_per_class=120):
    """Generate mock dataset split so that train.py runs out of the box."""
    logger.info("Generating mock eye images for instant runnability...")
    
    # Distribution: 70% Train, 15% Val, 15% Test
    train_count = int(samples_per_class * 0.70)
    val_count = int(samples_per_class * 0.15)
    test_count = samples_per_class - train_count - val_count
    
    splits = {
        'train': train_count,
        'val': val_count,
        'test': test_count
    }
    
    for split, count in splits.items():
        # Open eyes
        for i in range(count):
            img = generate_synthetic_eye(is_closed=False)
            filename = os.path.join(base_dir, split, 'open_eyes', f"open_{i:04d}.jpg")
            cv2.imwrite(filename, img)
            
        # Closed eyes
        for i in range(count):
            img = generate_synthetic_eye(is_closed=True)
            filename = os.path.join(base_dir, split, 'closed_eyes', f"closed_{i:04d}.jpg")
            cv2.imwrite(filename, img)
            
    logger.info(f"Generated {samples_per_class} open and closed eye images distributed in data/")

if __name__ == '__main__':
    create_directory_structure()
    build_mock_dataset()
    print("Dataset structure and mock images generated successfully.")
