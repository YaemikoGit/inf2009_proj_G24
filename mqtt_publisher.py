# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import time
from datetime import datetime
import json
import base64
import cv2
from camera.headcount import detect_faces_and_pose, get_camera
from collections import deque
import threading

try:
    from sensors.light import get_light
    HAS_LIGHT = True
except ImportError:
    HAS_LIGHT = False
    print("Light sensor not available - skipping")

try:
    from sensors.temperature import get_temperature
    HAS_TEMP = True
except ImportError:
    HAS_TEMP = False
    print("Temperature sensor not available - skipping")
    
try:
    from microphone.actual_noise import get_sound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False
    print("Sound sensor not available - skipping")
    
client = mqtt.Client("Publisher")

TOPIC_TEMP = "sensors/temperature"
TOPIC_LIGHT = "sensors/light"
TOPIC_HEADCOUNT = "sensors/headcount"
TOPIC_SOUND = "sensors/sound"

ATTENTION_THRESHOLD = 0.50
LATEST_SENSORS = {
    "temperature": 24.0,
    "humidity": 50.0,
    "light": "On",
    "sound_label": "quiet",
    "sound_dbfs": -50.0
}

# Alert rate limiting
last_alert_times = {
    "temperature": 0,
    "light": 0,
    "sound": 0,
    "attention": 0
}
ALERT_COOLDOWN = 10  # seconds between same alert type

# Message queue for async publishing
publish_queue = deque(maxlen=100)
publish_lock = threading.Lock()

# Camera frame buffer
latest_camera_data = {
    "count": 0,
    "attentive": 0,
    "distracted": 0,
    "image": None,
    "timestamp": None
}
camera_data_lock = threading.Lock()

# Sound data buffer
latest_sound_data = {
    "rms": 0,
    "peak": 0,
    "variance": 0,
    "dBFS": -50,
    "label": "quiet"
}
sound_data_lock = threading.Lock()

############ SOUND PROCESSING THREAD ################
def sound_processing_thread():
    """Dedicated thread for sound processing - runs independently"""
    global latest_sound_data
    
    print("Sound thread started!")
    
    while True:
        try:
            if HAS_SOUND:
                sound_data = get_sound()
                
                if sound_data:
                    # Update shared data
                    with sound_data_lock:
                        latest_sound_data = sound_data
            
            time.sleep(1)  # Sample sound every 1 second
            
        except Exception as e:
            print(f"Sound processing error: {e}")
            time.sleep(1)

# Start sound processing thread
if HAS_SOUND:
    sound_thread = threading.Thread(target=sound_processing_thread, daemon=True)
    sound_thread.start()

def on_message(client, userdata, message):
    global ATTENTION_THRESHOLD
    if message.topic == "settings/attention_threshold":
        try:
            payload = json.loads(message.payload.decode())
            ATTENTION_THRESHOLD = float(payload.get("threshold", 0.5))
            print(f"Updated attention threshold to {ATTENTION_THRESHOLD}")
        except Exception as e:
            print("Failed to parse threshold", e)

client.on_message = on_message
client.connect("172.20.10.12", 1883)
client.subscribe("settings/attention_threshold")
client.loop_start()

# Background thread to publish messages asynchronously
def publish_worker():
    while True:
        try:
            with publish_lock:
                if publish_queue:
                    topic, payload = publish_queue.popleft()
                else:
                    time.sleep(0.01)
                    continue
            
            # Publish outside the lock
            client.publish(topic, payload, qos=0)
        except Exception as e:
            print(f"Publish error: {e}")
            time.sleep(0.1)

# Start background publisher
publisher_thread = threading.Thread(target=publish_worker, daemon=True)
publisher_thread.start()

def async_publish(topic, payload):
    """Add message to queue for async publishing"""

    with publish_lock:
        publish_queue.append((topic, json.dumps(payload)))

def check_and_publish_sensor_alert(topic_suffix, sensor_type, is_bad, message_detail):
    if is_bad:
        # Rate limit alerts
        now = time.time()
        if now - last_alert_times.get(sensor_type, 0) < ALERT_COOLDOWN:
            return
        last_alert_times[sensor_type] = now
        
        payload = {
            "sensor": sensor_type,
            "message": message_detail,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        async_publish(f"alerts/sensor_alerts/{topic_suffix}", payload)

############ for temperature ################
def publish_temperature():
    if not HAS_TEMP:
        return None
    
    # Try up to 3 times with delays
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = get_temperature()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            payload = {
                "status": "ok",
                "temperature": float(data['temperature']),
                "humidity": float(data['humidity']),
                "timestamp": timestamp
            }
            
            LATEST_SENSORS["temperature"] = payload["temperature"]
            LATEST_SENSORS["humidity"] = payload["humidity"]
            
            temp_val = payload["temperature"]
            hum_val = payload["humidity"]

            is_temp_bad = temp_val < 20 or temp_val > 28
            is_hum_bad = hum_val < 30 or hum_val > 70
            is_bad = is_temp_bad or is_hum_bad
            
            msg_parts = []
            if is_temp_bad:
                if temp_val < 20:
                    msg_parts.append(f"Temperature too low ({temp_val}C)")
                else:
                    msg_parts.append(f"Temperature too high ({temp_val}C)")
                    
            if is_hum_bad:
                if hum_val < 30:
                    msg_parts.append(f"Humidity too low ({hum_val}%)")
                else:
                    msg_parts.append(f"Humidity too high ({hum_val}%)")

            msg = " | ".join(msg_parts) if msg_parts else "Normal"
            
            check_and_publish_sensor_alert("temperature_alerts", "temperature", is_bad, msg)
            
            async_publish(TOPIC_TEMP, payload)
            return payload  # Success! Exit retry loop
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Not the last attempt, wait and retry
                time.sleep(0.5)  # Wait 500ms before retry
                continue
            else:
                # Last attempt failed
                # Only print error once every 10 failures to reduce spam
                if not hasattr(publish_temperature, 'error_count'):
                    publish_temperature.error_count = 0
                publish_temperature.error_count += 1
                
                if publish_temperature.error_count % 10 == 1:
                    print(f"Temperature sensor error after {max_retries} attempts: {e}")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                payload = {
                    "status": "error",
                    "message": "Temperature sensor not detected",
                    "timestamp": timestamp
                }
                async_publish(TOPIC_TEMP, payload)
                return payload

############ for light ################
def publish_light_status():
    if not HAS_LIGHT:
        return None
    try:
        light_status = get_light()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        payload = {
            "status": "ok", 
            "light": light_status,
            "timestamp": timestamp
        }
        
        LATEST_SENSORS["light"] = light_status
        is_light_bad = str(light_status).lower() in ["false", "off"]
        check_and_publish_sensor_alert("light_alerts", "light", is_light_bad, "Classroom lights are turned OFF")
        
    except Exception as e:
        print(f"Light sensor disconnected: {e}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "status": "error", 
            "message": "Sensor Disconnected",
            "timestamp": timestamp
        }
    
    async_publish(TOPIC_LIGHT, payload)
    return payload
    
# Cache the blame probabilities calculation
blame_cache = None
blame_cache_time = 0
BLAME_CACHE_TTL = 2  # seconds

def calculate_blame_probabilities():
    global blame_cache, blame_cache_time
    
    # Return cached result if still valid
    now = time.time()
    if blame_cache and (now - blame_cache_time) < BLAME_CACHE_TTL:
        return blame_cache
    
    temp_bad = LATEST_SENSORS["temperature"] < 20 or LATEST_SENSORS["temperature"] > 28
    hum_bad = LATEST_SENSORS["humidity"] < 30 or LATEST_SENSORS["humidity"] > 70
    light_bad = str(LATEST_SENSORS["light"]).lower() in ["false", "off"]
    sound_bad = LATEST_SENSORS["sound_label"].lower() in ["noisy", "loud"]
    
    bad_count = sum([temp_bad, hum_bad, light_bad, sound_bad])
    
    if bad_count > 0:
        maj_share = 0.8 / bad_count
        min_share = 0.15 / (4 - bad_count) if bad_count < 4 else 0
        others_prob = 0.05 if bad_count < 4 else 0.0
        result = {
            "temperature_prob": round(maj_share if temp_bad else min_share, 2),
            "humidity_prob": round(maj_share if hum_bad else min_share, 2),
            "light_prob": round(maj_share if light_bad else min_share, 2),
            "sound_prob": round(maj_share if sound_bad else min_share, 2),
            "others_prob": round(others_prob, 2)
        }
    else:
        # Distance calculation for when all sensors are "OK"
        temp_dist = min(abs(LATEST_SENSORS["temperature"] - 24) / 10.0, 1.0)
        hum_dist = min(abs(LATEST_SENSORS["humidity"] - 50) / 40.0, 1.0)
        light_dist = 0.1
        
        dbfs = LATEST_SENSORS.get("sound_dbfs", -50.0)
        variance = LATEST_SENSORS.get("sound_variance", 0.0)
        rms = LATEST_SENSORS.get("sound_rms", 0.0)
        peak = LATEST_SENSORS.get("sound_peak", 0.0)
        
        sound_dist = 0.0
        if variance > 0.0:
            if dbfs <= -22.92:
                if rms <= 0.03:
                    sound_dist = rms / 0.03
                    if peak <= 0.38:
                        sound_dist = (sound_dist + 1.0) / 2.0
                else: 
                    if peak > 0.38:
                        sound_dist = 0.38 / peak if peak > 0 else 0
            else:
                if peak <= 0.45:
                    sound_dist = peak / 0.45
                    
        others_dist = 0.8

        total = temp_dist + hum_dist + light_dist + sound_dist + others_dist
        if total <= 0: total = 1
        
        result = {
            "temperature_prob": round(temp_dist/total, 2),
            "humidity_prob": round(hum_dist/total, 2),
            "light_prob": round(light_dist/total, 2),
            "sound_prob": round(sound_dist/total, 2),
            "others_prob": round(others_dist/total, 2)
        }
    
    # Update cache
    blame_cache = result
    blame_cache_time = now
    return result

############ CAMERA PROCESSING THREAD ################
last_encoded_image = None
last_encoded_image_lock = threading.Lock()

def camera_processing_thread():
    """Dedicated thread for camera processing - runs independently"""
    global latest_camera_data, last_encoded_image
    
    cam = get_camera()  
    if cam is None:
        print("Failed to initialize camera")
        return
    
    frame_count = 0
    last_image_encode_time = 0
    IMAGE_ENCODE_INTERVAL = 1  # Encode image every 1 second
    
    print("Camera thread started!")
    
    while True:
        try:
            if not cam.isOpened():
                print("Camera disconnected, reconnecting...")
                cam = get_camera()
                if cam is None:
                    time.sleep(1)
                    continue

            success, frame = cam.read()
            if not success:
                print("Camera read failed")
                cam.release()
                cam = get_camera()
                time.sleep(0.1)
                continue
            
            # Always process for headcount/attention
            annotated_frame, attention = detect_faces_and_pose(frame, return_attention=True)
            count = attention["attentive"] + attention["distracted"]
            
            # Only encode image if faces are detected
            now = time.time()
            if count > 0 and now - last_image_encode_time >= IMAGE_ENCODE_INTERVAL:
                # Resize frame before encoding to reduce size
                h, w = annotated_frame.shape[:2]
                new_w = 480  # Reduce width
                new_h = int(h * (new_w / w))
                small_frame = cv2.resize(annotated_frame, (new_w, new_h))
                
                # Encode with moderate quality
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 65]
                _, buffer = cv2.imencode('.jpg', small_frame, encode_params)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Update the last encoded image
                with last_encoded_image_lock:
                    last_encoded_image = image_base64
                
                last_image_encode_time = now
            
            # Update shared data
            with camera_data_lock:
                with last_encoded_image_lock:
                    # Only keep image if we still have faces
                    current_image = last_encoded_image if count > 0 else None
                
                latest_camera_data = {
                    "count": count,
                    "attentive": attention["attentive"],
                    "distracted": attention["distracted"],
                    "image": current_image,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            frame_count += 1
            
            # Small delay to prevent maxing out CPU
            time.sleep(0.05)  # ~20 FPS processing
            
        except Exception as e:
            print(f"Camera processing error: {e}")
            time.sleep(0.1)

# Start camera processing thread
camera_thread = threading.Thread(target=camera_processing_thread, daemon=True)
camera_thread.start()

############ for camera ################
def publish_headcount():
    """Just publish the latest camera data - no processing here"""
    global last_alert_times
    
    # Get the latest processed data
    with camera_data_lock:
        data = latest_camera_data.copy()
    
    # Build payload - ALWAYS include image if available
    payload = {
        "count": data["count"],
        "attentive": data["attentive"],
        "distracted": data["distracted"],
        "timestamp": data["timestamp"]
    }
    
    # Always include image if we have one
    if data["image"]:
        payload["image"] = data["image"]
    
    async_publish(TOPIC_HEADCOUNT, payload)
    print(f"Headcount: {data['count']} | Attentive: {data['attentive']} Distracted: {data['distracted']}")

    # Check for attention alerts with rate limiting
    if data["count"] > 0:
        attention_rate = data["attentive"] / data["count"]
        if attention_rate < ATTENTION_THRESHOLD:
            now = time.time()
            # Rate limit attention alerts
            if now - last_alert_times.get("attention", 0) >= ALERT_COOLDOWN:
                last_alert_times["attention"] = now
                probs = calculate_blame_probabilities()
                alert_payload = {
                    "attention_rate": round(attention_rate, 2),
                    "threshold": ATTENTION_THRESHOLD,
                    "probabilities": probs,
                    "timestamp": data["timestamp"]
                }
                async_publish("alerts/attention_alerts", alert_payload)
                print(f"Alert: Attention dropped to {attention_rate:.2f}. Causes: {probs}")


############ for sound ################
def publish_sound():
    if not HAS_SOUND:
        return None
    
    # Get the latest sound data from background thread (NOT calling get_sound() directly!)
    with sound_data_lock:
        sound_data = latest_sound_data.copy()
    
    try:
        payload = sound_data.copy()
        
        # Convert all numeric values to Python floats
        for key in ["rms", "peak", "variance", "dBFS"]:
            if key in payload:
                payload[key] = float(payload[key])
        
        if "label" in payload:
            payload["label"] = str(payload["label"])
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload["status"] = "ok"
        payload["timestamp"] = timestamp

        LATEST_SENSORS["sound_label"] = payload.get("label", "quiet")
        LATEST_SENSORS["sound_dbfs"] = payload.get("dBFS", -50.0)
        LATEST_SENSORS["sound_variance"] = payload.get("variance", 0.0)
        LATEST_SENSORS["sound_peak"] = payload.get("peak", 0.0)
        LATEST_SENSORS["sound_rms"] = payload.get("rms", 0.0)
        
        is_sound_bad = LATEST_SENSORS["sound_label"].lower() in ["noisy", "loud"]
        check_and_publish_sensor_alert("sound_alerts", "sound", is_sound_bad, f"Noise level is high ({LATEST_SENSORS['sound_label']})")
        
    except Exception as e:
        print(f"Sound sensor error: {e}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "status": "error",
            "message": "Sound sensor not detected",
            "timestamp": timestamp
        }
    
    async_publish(TOPIC_SOUND, payload)
    return payload

# Main loop to publish data at intervals
if __name__ == "__main__":
    # Wait for threads to initialize
    print("Waiting for threads to initialize...")
    time.sleep(2)
    
    # Initialize to current time so nothing fires immediately
    start_time = time.time()
    last_temp_time = start_time
    last_light_time = start_time
    last_sound_time = start_time
    last_headcount_time = start_time
    
    print("Starting sensor publisher...")
    print(f"Temperature: {'ENABLED' if HAS_TEMP else 'DISABLED'}")
    print(f"Light: {'ENABLED' if HAS_LIGHT else 'DISABLED'}")
    print(f"Sound: {'ENABLED' if HAS_SOUND else 'DISABLED'}")
    print(f"Camera: ENABLED")
    
    try:
        while True:
            now = time.time()
            
            # Temperature every 10 seconds
            if HAS_TEMP and now - last_temp_time >= 10:
                publish_temperature()
                last_temp_time = now
            
            # Light every 5 seconds
            if HAS_LIGHT and now - last_light_time >= 5:
                publish_light_status()
                last_light_time = now

            # Sound every 1 second
            if HAS_SOUND and now - last_sound_time >= 1:
                publish_sound()
                last_sound_time = now
            
            # Headcount every 1 second
            if now - last_headcount_time >= 1:
                publish_headcount()
                last_headcount_time = now
            
            time.sleep(0.1)
            
    finally:
        print("Camera released")
