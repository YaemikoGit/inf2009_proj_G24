import cv2
import numpy as np
import os
import glob
import time
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
LBF_path = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")
facemark.loadModel(LBF_path)

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
            time.sleep(1)  # give the USB a moment to initialize
    return camera
    
frame_counter = 0

# Smoothing buffers for pitch and yaw
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

# Function to detect faces and landmarks
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
            # 1. Convert face_boxes to the format Facemark expects (a list of rectangles)
            # Ensure face_boxes are [x, y, w, h]
            padding = 20
            formatted_boxes = []
            for box in face_boxes:
                x1, y1, x2, y2 = box
                formatted_boxes.append((x1, y1, x2 - x1, y2 - y1))

            # 2. Fit landmarks using the FULL frame and the boxes
            success, landmarks_all = facemark.fit(frame, np.array(formatted_boxes))

            if success:
                for i, landmarks in enumerate(landmarks_all):
                    # landmarks[0] contains the 68 points for the i-th face
                    points = landmarks[0] 

                    # 3. Map the 6 points for SolvePnP
                    
                    image_points = np.array([
                        points[30], # Nose tip
                        points[33], # Nose base (More stable than chin)
                        points[36], # Left Eye inner
                        points[45], # Right Eye inner
                        points[48], # Left Mouth corner
                        points[54]  # Right Mouth corner
                    ], dtype="double")

                    # --- DEBUG: Draw the points to verify they are ON the face ---
                    for (px, py) in image_points:
                        cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 255), -1)


                    model_points = np.array([
                        (0.0, 0.0, 0.0),           # Nose tip
                        (0.0, -50.0, 20.0),        # Nose base
                        (-30.0, 40.0, 10.0),       # Left Eye inner
                        (30.0, 40.0, 10.0),        # Right Eye inner
                        (-40.0, -60.0, 10.0),      # Left Mouth corner
                        (40.0, -60.0, 10.0)        # Right Mouth corner
                    ], dtype="double")

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
                            
                        center_yaw = 16.0
                        center_pitch = 12.0 

                        # Expand the thresholds to be more forgiving
                        yaw_threshold = 80.0   # Allows Yaw to be anywhere from -24 to 56
                        pitch_threshold = 80.0 # Allows Pitch to be anywhere from -28 to 52

                        # Use smoothed values
                        s_yaw = get_smoothed_yaw(yaw)
                        s_pitch = get_smoothed_pitch(pitch)

                        # Calculate difference
                        diff_yaw = abs(s_yaw - center_yaw)
                        diff_pitch = abs(s_pitch - center_pitch)
                        
                        #print(f"Debug: Yaw={diff_yaw:.2f}, Pitch={diff_pitch:.2f}, Success={success_pnp}")
                        # Determine state
                        is_attentive = diff_yaw < yaw_threshold and diff_pitch < pitch_threshold

                        if is_attentive:
                            attentive += 1
                            label, color = "Attentive", (0, 255, 0)
                        else:
                            distracted += 1
                            label, color = "Distracted", (0, 0, 255)
                            

                        # Draw Box and Label
                        x1, y1, x2, y2 = face_boxes[i]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"Y:{int(yaw)} P:{int(pitch)} R:{int(roll)}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

            frame = detect_faces_and_pose(frame)  # unchanged

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print("Error:", e)
            continue
