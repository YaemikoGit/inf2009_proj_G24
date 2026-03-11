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

# Create Facemark LBF
facemark = cv2.face.createFacemarkLBF()
# Load the model
model_path = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")
facemark.loadModel(model_path)

camera = cv2.VideoCapture(0)

frame_counter = 0


# Function to detect faces and landmarks
# def detect_faces_and_pose(frame, return_attention=False):  # <- added return_attention
#     h, w = frame.shape[:2]
#     headcount = 0
#     attentive = 0   # <- added
#     distracted = 0  # <- added
    
#     try:
#         # ---------- 1) SSD detection ----------
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
#                                      (104.0, 177.0, 123.0))
#         ssd_net.setInput(blob)
#         detections = ssd_net.forward()

#         face_boxes = []
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.5:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (x1, y1, x2, y2) = box.astype(int)
#                 face_boxes.append((x1, y1, x2, y2))

#         headcount = len(face_boxes)

#         # ---------- 2) landmarks + pose ----------
#         for (x1, y1, x2, y2) in face_boxes:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             face = frame[y1:y2, x1:x2]
#             if face.size == 0:
#                 continue

#             face_h, face_w = face.shape[:2]
#             resized = cv2.resize(face, (320, 320))
#             yunet.setInputSize((320, 320))
#             _, faces = yunet.detect(resized)

#             if faces is None:
#                 continue

#             face_data = faces[0]
#             if len(face_data) < 15:
#                 continue

#             landmarks = face_data[5:15].reshape((5, 2))
#             scale_x = face_w / 320
#             scale_y = face_h / 320

#             image_points = []
#             for (lx, ly) in landmarks:
#                 x = lx * scale_x + x1
#                 y = ly * scale_y + y1
#                 image_points.append([x, y])

#             image_points = np.array(image_points, dtype=np.float32)
#             if image_points.shape != (5, 2):
#                 continue
                
#             model_points = np.array([
#                 (-30.0,  40.0,  30.0),
#                 ( 30.0,  40.0,  30.0),
#                 (  0.0,   0.0,   0.0),
#                 (-25.0, -30.0,  20.0),
#                 ( 25.0, -30.0,  20.0)
#             ], dtype=np.float32)

#             focal_length = w
#             center = (w/2, h/2)
#             camera_matrix = np.array([
#                 [focal_length, 0, center[0]],
#                 [0, focal_length, center[1]],
#                 [0, 0, 1]
#             ], dtype=np.float32)

#             dist_coeffs = np.zeros((4, 1))

#             success, rot_vec, trans_vec = cv2.solvePnP(
#                 model_points, image_points, camera_matrix, dist_coeffs,
#                 flags=cv2.SOLVEPNP_EPNP
#             )

#             if not success:
#                 continue
                
#             rot_mat, _ = cv2.Rodrigues(rot_vec)
#             pose_mat = cv2.hconcat((rot_mat, trans_vec))
#             _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

#             pitch, yaw, roll = euler.flatten()
#             pitch = float(pitch)
#             yaw = float(yaw)

#             # Attention logic
#             # is_attentive = abs(yaw) < 25 and abs(pitch) < 20
#             # if is_attentive:
#             #     attentive += 1
#             #     label = "Attentive"
#             #     color = (0, 255, 0)   # green
#             # else:
#             #     distracted += 1
#             #     label = "Distracted"
#             #     color = (0, 0, 255)   # red

#             # cv2.putText(frame, f"{label} Y:{yaw:.1f} P:{pitch:.1f}",
#             #             (x1, y1 - 10),
#             #             cv2.FONT_HERSHEY_SIMPLEX,
#             #             0.5, color, 2)
    
#             # Yaw:
#             #  0° = facing camera
#             # + or - 25° = acceptable slight turn        
#             is_facing_forward = abs(yaw) < 25

#             # Pitch:
#             #  0° = forward
#             # -30° = looking slightly down (laptop)
#             # -40° = too far down -> distracted
#             # +20° = too far up -> distracted
#             is_laptop_or_forward = -35 < pitch < 15

#             is_attentive = is_facing_forward and is_laptop_or_forward

#             if is_attentive:
#                 attentive += 1
#                 label = "Attentive"
#                 color = (0, 255, 0)   # green
#             else:
#                 distracted += 1
#                 label = "Distracted"
#                 color = (0, 0, 255)   # red

#             cv2.putText(frame, f"{label} Y:{yaw:.1f} P:{pitch:.1f}",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5, color, 2)

#     except Exception as e:
#         print("Pose Error:", e)

#     # Always draw headcount
#     cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     if return_attention:
#         return frame, {"attentive": attentive, "distracted": distracted}
#     return frame

def detect_faces_and_pose(frame, return_attention=False):
    h, w = frame.shape[:2]
    headcount = 0
    attentive = 0
    distracted = 0
    
    try:
        # ---------- Step 1: Detect Faces with SSD ----------
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        ssd_net.setInput(blob)
        detections = ssd_net.forward()

        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_boxes.append(box.astype(int))

        headcount = len(face_boxes)

        if headcount > 0:
            # ---------- Step 2: Get 68 Landmarks ----------
            # facemark.fit requires a list of rectangles
            rects = [tuple(box) for box in face_boxes]
            success, landmarks_all = facemark.fit(frame, np.array(face_boxes))

            if success:
                for i, landmarks in enumerate(landmarks_all):
                    points = landmarks[0] # The 68 points
                    
                    # Selection of points for SolvePnP (Nose, Chin, Eyes, Mouth corners)
                    # Indices based on the 68-point map
                    image_points = np.array([
                        points[30],     # Nose tip
                        points[8],      # Chin
                        points[36],     # Left eye left corner
                        points[45],     # Right eye right corner
                        points[48],     # Left Mouth corner
                        points[54]      # Right mouth corner
                    ], dtype="double")

                    # Standard 3D model points (generic human face)
                    model_points = np.array([
                        (0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0),        # Chin
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corner
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0)      # Right mouth corner
                    ])

                    # Camera internals
                    focal_length = w
                    center = (w/2, h/2)
                    camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double"
                    )

                    dist_coeffs = np.zeros((4,1)) 
                    (success_pnp, rot_vec, trans_vec) = cv2.solvePnP(
                        model_points, image_points, camera_matrix, dist_coeffs, 
                        flags=cv2.SOLVEPNP_ITERATIVE # More accurate than EPNP
                    )

                    if success_pnp:
                        # Decompose to Euler angles
                        rmat, _ = cv2.Rodrigues(rot_vec)
                        pmat = cv2.hconcat((rmat, trans_vec))
                        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pmat)
                        
                        pitch, yaw, roll = euler.flatten()

                        # --- Attention Logic ---
                        # Tightened for better accuracy
                        is_facing_forward = abs(yaw) < 30 
                        is_looking_at_screen = -50 < pitch < 25 

                        is_attentive = is_facing_forward and is_looking_at_screen
                        if is_attentive:
                            attentive += 1
                            label, color = "Attentive", (0, 255, 0)
                        else:
                            distracted += 1
                            label, color = "Distracted", (0, 0, 255)

                        # Draw Box and Label
                        x1, y1, x2, y2 = face_boxes[i]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} Y:{int(yaw)} P:{int(pitch)}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    except Exception as e:
        print("Pose Error:", e)

    # UI Overlays
    cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if return_attention:
        return frame, {"attentive": attentive, "distracted": distracted}
    return frame


# FUnction to generate the frames
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
