import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print(f"✅ TensorFlow can access the GPU.")
    print(f"GPU devices found: {gpu_devices}")
else:
    print("❌ TensorFlow cannot access the GPU. Check your installation.")



IMAGES_PATH = os.path.join('data','images')
number_of_images = 30

cap = cv2.VideoCapture(0)
    
for imgnum in range(number_of_images):
    print('Collecting image{}',format(imgnum))
    ret,frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

images = tf.data.Dataset.list_files('data\\images\\*.jpg')
images.as_numpy_iterator().next()

def load_image(x):
    byte_img = tf.io.read_file(x)
    img= tf.image.decode_jpeg(byte_img)
    return img
images = images.map(load_image)
images.as_numpy_iterator().next()
type(images)

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx,image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()

for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data',folder, 'images')):
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data', 'labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath=os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath,new_filepath)
            
