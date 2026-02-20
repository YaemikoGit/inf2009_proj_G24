#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

echo "Downloading face detector model..."

wget -O models/res10_300x300_ssd_iter_140000.caffemodel \
https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

wget -O models/deploy.prototxt \
https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -O models/face_landmark_1000.onnx \
https://github.com/opencv/opencv_zoo/raw/master/models/face_landmark/face_landmark_1000.onnx

wget -O models/face_detection_yunet_2023mar.onnx \
https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

echo "Download complete."