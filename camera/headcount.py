import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load SSD face detector (long-distance reliable)
prototxt_path = os.path.join(BASE_DIR, "models", "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load YuNet
yunet_path = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
yunet = cv2.FaceDetectorYN.create(
    yunet_path, "", (320, 320)
)

# Load face landmarks
landmark_path = os.path.join(BASE_DIR, "models", "face_landmark_1000.onnx")
landmark_model = cv2.FaceDetectorYN.create(
    landmark_path, "", (320, 320)
)

camera = cv2.VideoCapture(0)

# 3D model points for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),            # nose
    (0.0, -330.0, -65.0),       # chin
    (-225.0, 170.0, -135.0),    # left eye
    (225.0, 170.0, -135.0),     # right eye
    (-150.0, -150.0, -125.0),   # left mouth
    (150.0, -150.0, -125.0)     # right mouth
], dtype=np.float32)


def detect_faces_and_pose(frame):
    h, w = frame.shape[:2]

    # ---------- 1) DNN FACE DETECTION ----------
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype(int)
        face_boxes.append((x1, y1, x2, y2))

    headcount = len(face_boxes)

    # ---------- 2) HEAD POSE ----------
    for (x1, y1, x2, y2) in face_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # Resize ROI for landmarks
        resized = cv2.resize(face_roi, (320, 320))

        _, lm = landmark_model.detect(resized)
        if lm is None:
            continue

        # Extract 6 landmarks
        lm = lm[0][:, :2]

        image_points = np.array([
            lm[0],  # nose
            lm[4],  # chin
            lm[1],  # left eye
            lm[2],  # right eye
            lm[3],  # left mouth
            lm[5],  # right mouth
        ], dtype=np.float32)

        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ])

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        if success:
            rot_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rot_mat, translation_vec))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

            yaw = euler[1][0]
            pitch = euler[0][0]

            cv2.putText(
                frame,
                f"Y:{yaw:.1f} P:{pitch:.1f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )

    cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = detect_faces_and_pose(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')