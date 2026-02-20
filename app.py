from flask import Flask, render_template, Response
import cv2, os
import numpy as np

# Load the DNN face detector
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

prototxt_path = os.path.join(BASE_DIR, "models", "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Detect ALL faces (multi-scale)
#         faces = face_cascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30)
#         )

#         # Draw a green rectangle around every detected face
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Add headcount text
#         count = len(faces)
#         cv2.putText(frame, f"Headcount: {count}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#         # Stream frame
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_faces_dnn(frame):
    h, w = frame.shape[:2]

    # Prepare image as DNN blob
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    count = 0

    # Loop through all detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Minimum confidence threshold
        if confidence < 0.5:
            continue

        count += 1

        # Extract bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Add headcount text
    cv2.putText(frame, f"Headcount: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run DNN headcount detection
        frame = detect_faces_dnn(frame)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)