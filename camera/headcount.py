import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

prototxt_path = os.path.join(BASE_DIR, "models", "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

camera = cv2.VideoCapture(0)


def detect_faces_dnn(frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:
            continue

        count += 1

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Headcount: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = detect_faces_dnn(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')