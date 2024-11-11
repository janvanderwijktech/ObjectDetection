# ObjectDetection
Python Object detection, with TensorFlow

EfficientDet Real-Time Object Detection on IP Camera Stream
This project demonstrates real-time object detection using TensorFlow’s EfficientDet model on a live video stream from an IP camera. The script loads a pre-trained EfficientDet model, captures video frames from an IP camera, and processes each frame to detect objects, displaying the results with bounding boxes around detected objects.

Requirements
TensorFlow: Ensure TensorFlow is installed (preferably tensorflow>=2.x) as it includes necessary functions for loading and running the EfficientDet model.
OpenCV: This project uses OpenCV for video capturing and displaying the object detection results.
EfficientDet Model: Pre-trained EfficientDet model (using efficientdet_d0_coco17_tpu-32/saved_model).
Usage
Load the Model: The script loads a pre-trained EfficientDet model stored in efficientdet_d0_coco17_tpu-32/saved_model.
Connect to IP Camera: Set the IP camera stream URL (replace url with the correct camera URL if different from the example in the script).
Run Object Detection: The script reads frames from the camera, preprocesses each frame for EfficientDet input, and runs object detection.
Display Output: Each frame is displayed with real-time object detection (bounding boxes around objects if processed).
Press 'q' to quit the video stream display.

Example

# Run the script
```python efficientdet_ip_camera.py```

Notes
Ensure the IP camera URL is accessible and formatted correctly.
Modify the section for bounding box drawing if additional customization is required for visualizing detections.
For continuous monitoring, consider optimizing frame capture and processing to enhance performance.

Requirements:

```http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz```

```https://github.com/chuanqi305/MobileNet-SSD/raw/refs/heads/master/mobilenet_iter_73000.caffemodel```
Save as:
```mobilenet.caffemodel```
