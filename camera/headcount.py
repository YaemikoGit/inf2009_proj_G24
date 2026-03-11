import cv2
import numpy as np
import os
#import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SSD face detector (long-distance)
prototxt_path = os.path.join(BASE_DIR, "models", "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
ssd_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# YuNet face detector + landmarks
yunet_model = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
yunet = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320))

# Mediapipe face landmarker
# mp_face = mp.solutions.face_mesh

# # Initialize model (Raspberry Pi optimized)
# face_mesh = mp_face.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=5,
#     refine_landmarks=True,   # gives iris + better head pose
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

camera = cv2.VideoCapture(0)

frame_counter = 0

def detect_faces_and_pose(frame, return_attention=False):  # <- added return_attention
    h, w = frame.shape[:2]
    headcount = 0
    attentive = 0   # <- added
    distracted = 0  # <- added
    
    try:
        # ---------- 1) SSD detection ----------
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

        # ---------- 2) landmarks + pose ----------
        for (x1, y1, x2, y2) in face_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_h, face_w = face.shape[:2]
            resized = cv2.resize(face, (320, 320))
            yunet.setInputSize((320, 320))
            _, faces = yunet.detect(resized)

            if faces is None:
                continue

            face_data = faces[0]
            if len(face_data) < 15:
                continue

            landmarks = face_data[5:15].reshape((5, 2))
            scale_x = face_w / 320
            scale_y = face_h / 320

            image_points = []
            for (lx, ly) in landmarks:
                x = lx * scale_x + x1
                y = ly * scale_y + y1
                image_points.append([x, y])

            image_points = np.array(image_points, dtype=np.float32)
            if image_points.shape != (5, 2):
                continue
                
            model_points = np.array([
                (-30.0,  40.0,  30.0),
                ( 30.0,  40.0,  30.0),
                (  0.0,   0.0,   0.0),
                (-25.0, -30.0,  20.0),
                ( 25.0, -30.0,  20.0)
            ], dtype=np.float32)

            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )

            if not success:
                continue
                
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            pose_mat = cv2.hconcat((rot_mat, trans_vec))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

            pitch, yaw, roll = euler.flatten()
            pitch = float(pitch)
            yaw = float(yaw)

            # Attention logic
            # is_attentive = abs(yaw) < 25 and abs(pitch) < 20
            # if is_attentive:
            #     attentive += 1
            #     label = "Attentive"
            #     color = (0, 255, 0)   # green
            # else:
            #     distracted += 1
            #     label = "Distracted"
            #     color = (0, 0, 255)   # red

            # cv2.putText(frame, f"{label} Y:{yaw:.1f} P:{pitch:.1f}",
            #             (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, color, 2)
    
            # Yaw:
            #  0° = facing camera
            # + or - 25° = acceptable slight turn        
            is_facing_forward = abs(yaw) < 25

            # Pitch:
            #  0° = forward
            # -30° = looking slightly down (laptop)
            # -40° = too far down -> distracted
            # +20° = too far up -> distracted
            is_laptop_or_forward = -35 < pitch < 15

            is_attentive = is_facing_forward and is_laptop_or_forward

            if is_attentive:
                attentive += 1
                label = "Attentive"
                color = (0, 255, 0)   # green
            else:
                distracted += 1
                label = "Distracted"
                color = (0, 0, 255)   # red

            cv2.putText(frame, f"{label} Y:{yaw:.1f} P:{pitch:.1f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    except Exception as e:
        print("Pose Error:", e)

    # Always draw headcount
    cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if return_attention:
        return frame, {"attentive": attentive, "distracted": distracted}
    return frame


# def detect_faces_and_pose(frame, return_attention=False):
#     h, w = frame.shape[:2]

#     # Run Mediapipe
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(frame_rgb)

#     headcount = 0
#     attentive = 0
#     distracted = 0

#     if results.multi_face_landmarks:
#         headcount = len(results.multi_face_landmarks)

#         for face in results.multi_face_landmarks:
#             # Get key points for head pose:
#             # Nose tip, left eye, right eye, mouth corners
#             lm = face.landmark

#             # Convert to pixel coords
#             def lp(i):
#                 return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

#             nose      = lp(1)
#             left_eye  = lp(33)
#             right_eye = lp(263)
#             mouth_l   = lp(61)
#             mouth_r   = lp(291)

#             image_points = np.array([
#                 nose, left_eye, right_eye, mouth_l, mouth_r
#             ], dtype=np.float32)

#             model_points = np.array([
#                 [0.0, 0.0, 0.0],      # Nose tip
#                 [-30.0, 30.0, -30.0], # Left eye
#                 [30.0, 30.0, -30.0],  # Right eye
#                 [-40.0, -30.0, -30.0],# Mouth left
#                 [40.0, -30.0, -30.0], # Mouth right
#             ], dtype=np.float32)

#             focal_length = w
#             center = (w / 2, h / 2)
#             camera_matrix = np.array([
#                 [focal_length, 0, center[0]],
#                 [0, focal_length, center[1]],
#                 [0, 0, 1]
#             ], dtype=np.float32)

#             dist_coeffs = np.zeros((4, 1))

#             success, rvec, tvec = cv2.solvePnP(
#                 model_points,
#                 image_points,
#                 camera_matrix,
#                 dist_coeffs,
#                 flags=cv2.SOLVEPNP_ITERATIVE
#             )

#             if not success:
#                 continue

#             rot_mat, _ = cv2.Rodrigues(rvec)
#             pose_mat = np.hstack((rot_mat, tvec))
#             _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

#             pitch = float(euler[0])
#             yaw   = float(euler[1])

#             # --- Attention logic ---
#             # Facing professor = yaw small
#             is_forward = abs(yaw) < 25       # Facing camera
#             is_laptop  = -40 < pitch < 10    # Looking slightly downward

#             is_attentive = is_forward or is_laptop

#             label = "Attentive" if is_attentive else "Distracted"
#             color = (0, 255, 0) if is_attentive else (0, 0, 255)

#             if is_attentive:
#                 attentive += 1
#             else:
#                 distracted += 1

#             cv2.putText(
#                 frame,
#                 f"{label} Y:{yaw:.1f} P:{pitch:.1f}",
#                 (int(nose[0]), int(nose[1] - 20)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 color,
#                 2
#             )

#     # Draw headcount on frame
#     cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     if return_attention:
#         return frame, {
#             "attentive": attentive,
#             "distracted": distracted
#         }

#     return frame

def generate_frames():
    while True:
        try:
            success, frame = camera.read()
            if not success:
                break

            frame = detect_faces_and_pose(frame)  # unchanged

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print("Error:", e)
            continue
