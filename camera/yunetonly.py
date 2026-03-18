import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# YuNet face detector + landmarks
yunet_model = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
yunet = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320))

camera = cv2.VideoCapture(0)
# Lower resolution for better performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0
POSE_ANALYSIS_INTERVAL = 3  # Analyze pose every 3 frames

# Smoothing buffers for pitch and yaw
pitch_history = []
yaw_history = []

# Cache for last detected state
last_face_boxes = []
last_attention_state = []  # List of (label, color, yaw, pitch, roll) for each face

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
    global frame_counter, last_face_boxes, last_attention_state
    
    h, w = frame.shape[:2]
    headcount = 0
    attentive = 0
    distracted = 0
    
    # Increment frame counter
    frame_counter += 1
    should_analyze_pose = (frame_counter % POSE_ANALYSIS_INTERVAL == 0)
    
    try:
        # Set input size for YuNet
        yunet.setInputSize((w, h))
        _, yunet_faces = yunet.detect(frame)

        face_boxes = []
        
        # Extract face boxes from YuNet detections
        if yunet_faces is not None:
            for yunet_face in yunet_faces:
                x, y, w_box, h_box = yunet_face[:4].astype(int)
                x1, y1 = x, y
                x2, y2 = x + w_box, y + h_box
                face_boxes.append([x1, y1, x2, y2])

        headcount = len(face_boxes)

        # ---------- Step 2: Analyze pose (every 3 frames) ----------
        if should_analyze_pose and headcount > 0:
            # Update cache
            last_face_boxes = face_boxes.copy()
            last_attention_state = []
            
            # Process each YuNet detection
            for i, yunet_face in enumerate(yunet_faces):
                # Extract landmarks directly from YuNet
                right_eye = yunet_face[4:6]
                left_eye = yunet_face[6:8]
                nose_tip = yunet_face[8:10]
                right_mouth = yunet_face[10:12]
                left_mouth = yunet_face[12:14]
                
                # Map to 6 points for SolvePnP
                image_points = np.array([
                    nose_tip,      # Nose tip
                    nose_tip,      # Nose base (use nose tip)
                    left_eye,      # Left Eye inner
                    right_eye,     # Right Eye inner
                    left_mouth,    # Left Mouth corner
                    right_mouth    # Right Mouth corner
                ], dtype="double")

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
                    flags=cv2.SOLVEPNP_ITERATIVE
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
                    yaw_threshold = 80.0
                    pitch_threshold = 80.0

                    # Use smoothed values
                    s_yaw = get_smoothed_yaw(yaw)
                    s_pitch = get_smoothed_pitch(pitch)

                    # Calculate difference
                    diff_yaw = abs(s_yaw - center_yaw)
                    diff_pitch = abs(s_pitch - center_pitch)
                    
                    # Determine state
                    is_attentive = diff_yaw < yaw_threshold and diff_pitch < pitch_threshold
                    if is_attentive:
                        attentive += 1
                        label, color = "Attentive", (0, 255, 0)
                    else:
                        distracted += 1
                        label, color = "Distracted", (0, 0, 255)
                    
                    # Cache the state
                    last_attention_state.append((label, color, yaw, pitch, roll))
                else:
                    # solvePnP failed, mark as unknown
                    last_attention_state.append(("Detecting...", (255, 255, 0), 0, 0, 0))
            
            # Draw boxes for analyzed faces
            for i, box in enumerate(last_face_boxes):
                if i < len(last_attention_state):
                    label, color, yaw, pitch, roll = last_attention_state[i]
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Y:{int(yaw)} P:{int(pitch)} R:{int(roll)}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        elif headcount > 0 and len(last_attention_state) > 0:
            # Use cached state from previous analysis
            for i, box in enumerate(face_boxes):
                if i < len(last_attention_state):
                    label, color, yaw, pitch, roll = last_attention_state[i]
                    
                    # Count for return_attention
                    if label == "Attentive":
                        attentive += 1
                    elif label == "Distracted":
                        distracted += 1
                    
                    # Draw with cached state
                    x1, y1, x2, y2 = box
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

# Function to generate the frames
def generate_frames():
    while True:
        try:
            success, frame = camera.read()
            if not success:
                break

            frame = detect_faces_and_pose(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print("Error:", e)
            continue
