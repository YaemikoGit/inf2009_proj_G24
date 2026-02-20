import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SSD face detector (long-distance)
prototxt_path = os.path.join(BASE_DIR, "models", "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
ssd_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# YuNet face detector + landmarks
yunet_model = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
yunet = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320))

camera = cv2.VideoCapture(0)

# 3D model points for solvePnP
model_points = np.array([
    (0.0, 0.0, 0.0),      # nose tip
    (0.0, -330.0, -65.0), # chin (approx)
    (-225.0, 170.0, -135.0), # left eye
    (225.0, 170.0, -135.0),  # right eye
    (-150.0, -150.0, -125.0), # left mouth
    (150.0, -150.0, -125.0)   # right mouth
], dtype=np.float32)


def detect_faces_and_pose(frame):
    h, w = frame.shape[:2]

    # --- STEP 1: SSD face detection ---
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)
            face_boxes.append((x1, y1, x2, y2))

    headcount = len(face_boxes)

    # --- STEP 2: Landmarks + Head Pose with YuNet ---
    for (x1, y1, x2, y2) in face_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Run YuNet for landmarks
        resized = cv2.resize(face, (320, 320))
        _, faces = yunet.detect(resized)
        if faces is None:
            continue

        # YuNet outputs 5 facial landmarks
        # shape: (N, 15) — xy pairs for 5 key points
        landmarks = faces[0].reshape(-1, 3)

        image_points = np.array([
            landmarks[0][:2], # left eye
            landmarks[1][:2], # right eye
            landmarks[2][:2], # nose tip
            landmarks[3][:2], # left mouth
            landmarks[4][:2]  # right mouth
        ], dtype=np.float32)

        # Camera intrinsics
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ])

        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        if success:
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            pose_mat = cv2.hconcat((rot_mat, trans_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            yaw = euler_angles[1][0]
            pitch = euler_angles[0][0]
            roll = euler_angles[2][0]

            cv2.putText(frame, f"Y:{yaw:.1f} P:{pitch:.1f}", 
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

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