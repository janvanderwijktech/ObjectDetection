import tensorflow as tf
import cv2

# Laad een EfficientDet-model
model = tf.saved_model.load('efficientdet_d0_coco17_tpu-32/saved_model')

# Open de IP-camera stream
url = 'http://192.168.2.80/mjpg/video.mjpg'
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Kan de videostream niet openen.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # EfficientDet vereist preprocessing (schaal/resize de image)
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Voer objectdetectie uit
    detections = model(input_tensor)

    # Verwerk en teken de resultaten
    # Hier zou je code toevoegen om bounding boxes te tekenen, vergelijkbaar met eerdere voorbeelden

    # Toon het frame
    cv2.imshow("EfficientDet Objectdetectie", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
