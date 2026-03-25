# import cv2
# import numpy as np
# import os
# import glob
# import time
# #import mediapipe as mp

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # SSD face detector (long-distance)
# prototxt_path = os.path.join(BASE_DIR, "models", "deploy.prototxt")
# model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
# ssd_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# # YuNet face detector + landmarks
# yunet_model = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
# yunet = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320))

# # Create Facemark LBF
# facemark = cv2.face.createFacemarkLBF()
# # Load the model
# LBF_path = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")
# facemark.loadModel(LBF_path)

# camera = None

# def find_working_camera():
#     for dev in glob.glob('/dev/video*'):
#         cap = cv2.VideoCapture(dev)
#         if cap.isOpened():
#             print(f"Camera found at {dev}")
#             return cap
#         cap.release()
#     return None

# def get_camera():
#     global camera
#     if camera is None or not camera.isOpened():
#         print("Camera disconnected. Scanning for new camera...")
#         camera = find_working_camera()
#         if camera is None:
#             time.sleep(1)  # give the USB a moment to initialize
#     return camera
    
# frame_counter = 0

# # Smoothing buffers for pitch and yaw
# pitch_history = []
# yaw_history = []

# def get_smoothed_pitch(new_val):
#     pitch_history.append(new_val)
#     if len(pitch_history) > 10: pitch_history.pop(0)
#     return sum(pitch_history) / len(pitch_history)

# def get_smoothed_yaw(new_val):
#     yaw_history.append(new_val)
#     if len(yaw_history) > 10: yaw_history.pop(0)
#     return sum(yaw_history) / len(yaw_history)

# # Function to detect faces and landmarks
# def detect_faces_and_pose(frame, return_attention=False):
#     h, w = frame.shape[:2]
#     headcount = 0
#     attentive = 0
#     distracted = 0

#     try:
#         # ---------- Step 1: Detect Faces with SSD ----------
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#         ssd_net.setInput(blob)
#         detections = ssd_net.forward()

#         face_boxes = []
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.5:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 face_boxes.append(box.astype(int))

#         headcount = len(face_boxes)

#         if headcount > 0:
#             # 1. Convert face_boxes to the format Facemark expects (a list of rectangles)
#             # Ensure face_boxes are [x, y, w, h]
#             padding = 20
#             formatted_boxes = []
#             for box in face_boxes:
#                 x1, y1, x2, y2 = box
#                 formatted_boxes.append((x1, y1, x2 - x1, y2 - y1))

#             # 2. Fit landmarks using the FULL frame and the boxes
#             success, landmarks_all = facemark.fit(frame, np.array(formatted_boxes))

#             if success:
#                 for i, landmarks in enumerate(landmarks_all):
#                     # landmarks[0] contains the 68 points for the i-th face
#                     points = landmarks[0] 

#                     # 3. Map the 6 points for SolvePnP
                    
#                     image_points = np.array([
#                         points[30], # Nose tip
#                         points[33], # Nose base (More stable than chin)
#                         points[36], # Left Eye inner
#                         points[45], # Right Eye inner
#                         points[48], # Left Mouth corner
#                         points[54]  # Right Mouth corner
#                     ], dtype="double")

#                     # --- DEBUG: Draw the points to verify they are ON the face ---
#                     for (px, py) in image_points:
#                         cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 255), -1)


#                     model_points = np.array([
#                         (0.0, 0.0, 0.0),           # Nose tip
#                         (0.0, -50.0, 20.0),        # Nose base
#                         (-30.0, 40.0, 10.0),       # Left Eye inner
#                         (30.0, 40.0, 10.0),        # Right Eye inner
#                         (-40.0, -60.0, 10.0),      # Left Mouth corner
#                         (40.0, -60.0, 10.0)        # Right Mouth corner
#                     ], dtype="double")

#                     # Camera internals
#                     focal_length = w
#                     center = (w/2, h/2)
#                     camera_matrix = np.array(
#                         [[focal_length, 0, center[0]],
#                          [0, focal_length, center[1]],
#                          [0, 0, 1]], dtype="double"
#                     )

#                     dist_coeffs = np.zeros((4,1)) 
#                     (success_pnp, rot_vec, trans_vec) = cv2.solvePnP(
#                         model_points, image_points, camera_matrix, dist_coeffs, 
#                         flags=cv2.SOLVEPNP_ITERATIVE # More accurate than EPNP
#                     )

#                     if success_pnp:
#                         # Decompose to Euler angles
#                         rmat, _ = cv2.Rodrigues(rot_vec)
#                         pmat = cv2.hconcat((rmat, trans_vec))
#                         _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pmat)
                        
                        
#                         pitch, yaw, roll = euler.flatten()
                            
#                         center_yaw = 16.0
#                         center_pitch = 12.0 

#                         # Expand the thresholds to be more forgiving
#                         yaw_threshold = 80.0   # Allows Yaw to be anywhere from -24 to 56
#                         pitch_threshold = 80.0 # Allows Pitch to be anywhere from -28 to 52

#                         # Use smoothed values
#                         s_yaw = get_smoothed_yaw(yaw)
#                         s_pitch = get_smoothed_pitch(pitch)

#                         # Calculate difference
#                         diff_yaw = abs(s_yaw - center_yaw)
#                         diff_pitch = abs(s_pitch - center_pitch)
                        
#                         #print(f"Debug: Yaw={diff_yaw:.2f}, Pitch={diff_pitch:.2f}, Success={success_pnp}")
#                         # Determine state
#                         is_attentive = diff_yaw < yaw_threshold and diff_pitch < pitch_threshold

#                         if is_attentive:
#                             attentive += 1
#                             label, color = "Attentive", (0, 255, 0)
#                         else:
#                             distracted += 1
#                             label, color = "Distracted", (0, 0, 255)
                            

#                         # Draw Box and Label
#                         x1, y1, x2, y2 = face_boxes[i]
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                         cv2.putText(frame, f"Y:{int(yaw)} P:{int(pitch)} R:{int(roll)}", (x1, y2 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     except Exception as e:
#         print("Pose Error:", e)

#     # UI Overlays
#     cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     if return_attention:
#         return frame, {"attentive": attentive, "distracted": distracted}
#     return frame


# # FUnction to generate the frames
# def generate_frames():
#     global camera 
#     while True:
#         try:
#             if camera is None or not camera.isOpened():
#                 camera = get_camera()
#                 if camera is None: 
#                     continue
                
#             success, frame = camera.read()
#             if not success:
#                 print("Camera read failed")
#                 camera.release()
#                 camera = None
#                 continue

#             frame = detect_faces_and_pose(frame)  # unchanged

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#         except Exception as e:
#             print("Error:", e)
#             continue


import cv2
import numpy as np
import os
import glob
import time
import math
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# YuNet face detector (face + 5 landmarks)
yunet_model = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
yunet = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320))


# Optional tuning
yunet.setScoreThreshold(0.6)
yunet.setNMSThreshold(0.3)

camera = None

def find_working_camera():
    for dev in glob.glob('/dev/video*'):
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            print(f"Camera found at {dev}")
            return cap
        cap.release()
    return None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        print("Camera disconnected. Scanning for new camera...")
        camera = find_working_camera()
        if camera is None:
            time.sleep(1)
    return camera



model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -110.0, -25.0),      # Chin
    (-75.0, 65.0, -50.0),      # Left eye
    (75.0, 65.0, -50.0),       # Right eye
    (-60.0, -50.0, -50.0),     # Left mouth corner
    (60.0, -50.0, -50.0)       # Right mouth corner
], dtype="double")

center_yaw = None
center_pitch = None
CENTERS_FILE = os.path.join(BASE_DIR, "pose_centers.json")

def load_centers():
    global center_yaw, center_pitch
    try:
        with open(CENTERS_FILE, "r") as f:
            d = json.load(f)
            center_yaw = float(d.get("yaw", 0.0))
            center_pitch = float(d.get("pitch", 0.0))
            print(f"Loaded centers yaw={center_yaw:.2f} pitch={center_pitch:.2f}")
    except Exception:
        center_yaw = None
        center_pitch = None
        print("No pose_centers.json found — run calibrate_pose() or allow one-shot auto-calibration.")

def save_centers(yaw_val, pitch_val):
    try:
        with open(CENTERS_FILE, "w") as f:
            json.dump({"yaw": float(yaw_val), "pitch": float(pitch_val)}, f)
            print(f"Saved centers yaw={yaw_val:.2f} pitch={pitch_val:.2f}")
    except Exception as e:
        print("Failed to save centers:", e)


# simple helper to extract yaw/pitch for the first detected face in a frame
def get_pose_values(frame):
    h, w = frame.shape[:2]
    try:
        yunet.setInputSize((w, h))
        ret, faces = yunet.detect(frame)
        if ret > 0 and faces is not None and len(faces) > 0:
            face = faces[0]
            x, y, w_box, h_box = face[:4].astype(int)

            lm = np.array(face[5:15], dtype=float).reshape((5, 2))
            # If Yunet returns normalized landmarks (0..1) scale to image
            if lm.max() <= 1.01:
                lm[:, 0] *= w
                lm[:, 1] *= h

            left_eye, right_eye, nose, left_mouth, right_mouth = lm
            chin = np.array([x + w_box / 2.0, y + h_box])
            image_points = np.array([nose, chin, left_eye, right_eye, left_mouth, right_mouth], dtype="double")
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0,0,1]], dtype="double")
            ok, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rmat, _ = cv2.Rodrigues(rot_vec)
                # Euler extraction (yaw, pitch, roll)
                sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
                # pitch
                pitch = math.degrees(math.atan2(-rmat[2,0], sy))
                # yaw
                yaw = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
                # roll
                roll = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
                return True, (yaw, pitch, roll)
    except Exception as e:
        print("get_pose_values error:", e)
    return False, (None, None, None)

# calibration routine: call once while looking straight at camera
def calibrate_pose(samples=50, delay=0.05):
    cam = get_camera()
    if cam is None:
        print("No camera for calibration")
        return
    vals_yaw = []
    vals_pitch = []
    collected = 0
    print("Calibration: please look straight at the camera for a few seconds...")
    while collected < samples:
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.05)
            continue
        ok, (yaw, pitch, _) = get_pose_values(frame)
        if ok and yaw is not None:
            vals_yaw.append(yaw)
            vals_pitch.append(pitch)
            collected += 1
        time.sleep(delay)
    if len(vals_yaw) > 0:
        mean_yaw = sum(vals_yaw) / len(vals_yaw)
        mean_pitch = sum(vals_pitch) / len(vals_pitch)
        save_centers(mean_yaw, mean_pitch)
        load_centers()
    else:
        print("Calibration failed: no valid samples")

# Load centers at module import
load_centers()

# Smoothing buffers
pitch_history = []
yaw_history = []

def get_smoothed_pitch(new_val):
    pitch_history.append(new_val)
    if len(pitch_history) > 10: pitch_history.pop(0)
    return sum(pitch_history) / len(pitch_history)

def get_smoothed_yaw(new_val):
    yaw_history.append(new_val)
    if len(yaw_history) > 10: yaw_history.pop(0)
    return sum(yaw_history) / len(yaw_history)

def detect_faces_and_pose(frame, return_attention=False):
    global center_yaw, center_pitch
    h, w = frame.shape[:2]
    headcount = 0
    attentive, distracted = 0, 0

    try:
        yunet.setInputSize((w, h))
        ret, faces = yunet.detect(frame)

        if ret > 0 and faces is not None:
            headcount = len(faces)
            for face in faces:
                x, y, w_box, h_box = face[:4].astype(int)

                # YuNet Landmarks: [5:15] represents 5 points (x,y)
                lm = np.array(face[5:15], dtype=float).reshape((5, 2))
                # If values are normalized, scale to pixel coords
                if lm.max() <= 1.01:
                    lm[:, 0] *= w
                    lm[:, 1] *= h

                # Correct Mapping (YuNet Order)
                l_eye   = lm[0]
                r_eye   = lm[1]
                nose    = lm[2]
                l_mouth = lm[3]
                r_mouth = lm[4]

                # Synthetic Chin (Bottom middle of the bounding box)
                chin = np.array([x + w_box / 2.0, y + h_box])

                # MUST match the order of your 'model_points' array
                image_points = np.array([
                    nose,       # 1. Nose tip
                    chin,       # 2. Chin
                    l_eye,      # 3. Left Eye
                    r_eye,      # 4. Right Eye
                    l_mouth,    # 5. Left Mouth
                    r_mouth     # 6. Right Mouth
                ], dtype="double")

                # --- Draw Landmarks (Validation) ---
                for i, pt in enumerate(image_points):
                    ix, iy = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= ix < w and 0 <= iy < h:
                        cv2.circle(frame, (ix, iy), 4, (0, 255, 255) if i == 1 else (255, 0, 255), -1)

                # --- SolvePnP ---
                focal_length = w
                cam_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype="double")
                ok, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)

                if ok:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
                    pitch = math.degrees(math.atan2(-rmat[2,0], sy))
                    yaw = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
                    roll = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))

                    s_yaw = get_smoothed_yaw(yaw)
                    s_pitch = get_smoothed_pitch(pitch)

                    # One-shot auto-calibration if centers not set
                    if center_yaw is None or center_pitch is None:
                        center_yaw = s_yaw
                        center_pitch = s_pitch
                        save_centers(center_yaw, center_pitch)
                        print(f"Auto-calibrated centers -> yaw:{center_yaw:.2f} pitch:{center_pitch:.2f}")

                    # --- Adaptive thresholds by face size (distance) ---
                    # helper to compute angular difference correctly (handles wrap-around)
                    def ang_diff(a, b):
                        d = (a - b + 180.0) % 360.0 - 180.0
                        return abs(d)

                    # face height in pixels (use for distance estimate)
                    face_h = float(h_box)
                    # reference face height when close to camera (tweak if needed)
                    REFERENCE_FACE_HEIGHT = 220.0
                    # scale >1 when face is smaller (farther) so thresholds relax
                    scale = max(0.6, min(3.0, REFERENCE_FACE_HEIGHT / max(20.0, face_h)))

                    # base thresholds (close-up)
                    BASE_YAW_THRESHOLD = 25.0
                    BASE_PITCH_THRESHOLD = 20.0

                    yaw_threshold = BASE_YAW_THRESHOLD * scale
                    pitch_threshold = BASE_PITCH_THRESHOLD * scale

                    # Debug prints (include thresholds)
                    print(f"DEBUG | Raw Yaw: {yaw:6.1f} | Raw Pitch: {pitch:6.1f} | Smooth Y: {s_yaw:6.1f} | Smooth P: {s_pitch:6.1f} | C_yaw: {center_yaw:.1f} C_pitch: {center_pitch:.1f} | YT:{yaw_threshold:.1f} PT:{pitch_threshold:.1f}")

                    # compute diffs using angular diff for yaw
                    diff_yaw = ang_diff(s_yaw, center_yaw)
                    diff_pitch = abs(s_pitch - center_pitch)

                    is_attentive = (diff_yaw < yaw_threshold) and (diff_pitch < pitch_threshold)

                    label = "Attentive" if is_attentive else "Distracted"
                    color = (0, 255, 0) if is_attentive else (0, 0, 255)

                    if is_attentive:
                        attentive += 1
                    else:
                        distracted += 1

                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                    cv2.putText(frame, f"{label} Y:{int(s_yaw)} P:{int(s_pitch)}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    except Exception as e:
        print(f"Pose Error: {e}")

    cv2.putText(frame, f"Headcount: {headcount}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if return_attention:
        return frame, {"attentive": attentive, "distracted": distracted}

    return frame

def generate_frames():
    global camera

    while True:
        try:
            if camera is None or not camera.isOpened():
                camera = get_camera()
                if camera is None:
                    continue

            success, frame = camera.read()

            if not success:
                print("Camera read failed")
                camera.release()
                camera = None
                continue

            frame = detect_faces_and_pose(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print("Error:", e)
            continue