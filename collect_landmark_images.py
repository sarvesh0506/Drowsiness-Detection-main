import cv2
import os
import uuid
import time

# --- SETUP ---
IMAGES_PATH = os.path.join('data', 'landmarks', 'images')
os.makedirs(IMAGES_PATH, exist_ok=True)
NUMBER_OF_IMAGES = 300 # Aim for 200-300 images

cap = cv2.VideoCapture(0)
img_count = 0

print(f"Ready to collect {NUMBER_OF_IMAGES} images.")
print("Press SPACEBAR to save an image. Press 'q' to quit.")

# --- COLLECTION LOOP ---
while cap.isOpened() and img_count < NUMBER_OF_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Display the image count on the screen
    cv2.putText(frame, f"Saved: {img_count}/{NUMBER_OF_IMAGES}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Image Collection', frame)

    key = cv2.waitKey(1) & 0xFF

    # Save frame on spacebar press
    if key == 32: # 32 is the ASCII code for spacebar
        img_name = os.path.join(IMAGES_PATH, f'{uuid.uuid1()}.jpg')
        cv2.imwrite(img_name, frame)
        img_count += 1
        print(f"Saved image {img_count}/{NUMBER_OF_IMAGES}")
        time.sleep(0.1) # Small delay to prevent multiple saves

    # Quit on 'q' press
    if key == ord('q'):
        break

print(f"Collection finished. Total images saved: {img_count}")
cap.release()
cv2.destroyAllWindows()