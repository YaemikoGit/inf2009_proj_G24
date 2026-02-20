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

    # ---------- 1) SSD detection (for long-range) ----------
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

    # ---------- 2) YuNet (landmarks + pose) ----------
    for (x1, y1, x2, y2) in face_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_h, face_w = face.shape[:2]

        # Resize for YuNet
        resized = cv2.resize(face, (320, 320))
        yunet.setInputSize((320, 320))
        _, faces = yunet.detect(resized)

        if faces is None:
            continue

        # YuNet returns 15 landmark values + face box
        # We only need the 5 landmark sets
        lm = faces[0][4:19].reshape(5, 3)  # 5 points, (x, y, score)

        # Scale back to original ROI (not 320x320)
        scale_x = face_w / 320
        scale_y = face_h / 320

        image_points = []
        for i in range(5):
            lx = lm[i][0] * scale_x + x1
            ly = lm[i][1] * scale_y + y1
            image_points.append([lx, ly])

        image_points = np.array(image_points, dtype=np.float32)

        # SAFETY CHECK: must have 5 landmarks
        if image_points.shape != (5, 2):
            continue

        # Use a matching 3D model (5 points)
        model_points_5 = np.array([
            (-30, 40, 30),   # left eye
            (30, 40, 30),    # right eye
            (0, 0, 0),       # nose tip
            (-25, -40, 20),  # left mouth
            (25, -40, 20)    # right mouth
        ], dtype=np.float32)

        focal_length = w
        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))

        # SAFETY CHECK: skip solvePnP if points invalid
        if image_points.shape[0] < 4:
            continue

        success, rot_vec, trans_vec = cv2.solvePnP(
            model_points_5,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        pose_mat = cv2.hconcat((rot_mat, trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

        yaw = float(euler[1])
        pitch = float(euler[0])

        cv2.putText(frame,
                    f"Y:{yaw:.1f} P:{pitch:.1f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2)

    cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print("YUface raw output:", faces[0])
    print("Length:", len(faces[0]))

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