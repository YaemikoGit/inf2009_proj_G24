import cv2
import numpy as np
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Load DNN SSD face detector (for robust long-distance detection)
dnn_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "deploy.prototxt"),
    os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
)

# Load YuNet:
yunet = cv2.FaceDetectorYN.create(
    os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx"),
    "",
    (320, 320)
)

# Face landmark model
landmark_model = cv2.FaceDetectorYN.create(
    os.path.join(MODEL_DIR, "face_landmark_1000.onnx"),
    "",
    (320, 320)
)

# 3D model points for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye
    (225.0, 170.0, -135.0),     # Right eye
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
], dtype=np.float32)


def detect_faces_and_pose(frame):

    h, w = frame.shape[:2]

    # DNN face detection for strong bounding boxes
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            face_boxes.append(box.astype(int))

    headcount = len(face_boxes)

    results = []
    for (x1, y1, x2, y2) in face_boxes:
        # Crop face region for landmark model
        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            continue

        # Prepare YuNet landmark detection
        face_resized = cv2.resize(face_roi, (320, 320))
        _, landmarks = landmark_model.detect(face_resized)

        if landmarks is None:
            continue

        # Extract 6 main points for solvePnP
        lm = landmarks[0][:, :2]

        image_points = np.array([
            lm[0],  # nose
            lm[4],  # chin
            lm[1],  # left eye
            lm[2],  # right eye
            lm[3],  # left mouth
            lm[5],  # right mouth
        ], dtype=np.float32)

        # Camera intrinsics
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ])

        dist_coeffs = np.zeros((4, 1))

        # Solve head pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        if success:
            # Convert to Euler angles (yaw, pitch, roll)
            rot_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rot_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            yaw = euler_angles[1][0]
            pitch = euler_angles[0][0]
            roll = euler_angles[2][0]

            results.append({
                "box": (x1, y1, x2, y2),
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            })

    return headcount, results