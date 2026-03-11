# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import time
from datetime import datetime
import json
import base64
import cv2
from camera.headcount import detect_faces_and_pose, ssd_net, camera

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

client = mqtt.Client("Publisher")
client.connect("192.168.137.42", 1883)

TOPIC_TEMP = "sensors/temperature"
TOPIC_LIGHT = "sensors/light"
TOPIC_HEADCOUNT = "sensors/headcount"

############ for temperature ################
def publish_temperature():
    if not HAS_TEMP:
        return None
    try:
        data = get_temperature()
        payload = {
            "status": "ok",
            "temperature": data['temperature'],
            "humidity": data['humidity']
        }
    except Exception as e:
        print(f"Temperature sensor error: {e}")
        payload = {"status": "error", "message": "Temperature sensor not detected"}
    
    client.publish(TOPIC_TEMP, json.dumps(payload))
    return payload

############ for light ################
def publish_light_status():
    if not HAS_LIGHT:
        return None
    try:
        light_status = get_light()
        payload = {"status": "ok", "light": light_status}
    except Exception as e:
        print(f"Light sensor disconnected: {e}")
        payload = {"status": "error", "message": "Sensor Disconnected"}
    
    client.publish(TOPIC_LIGHT, json.dumps(payload))
    return payload

############ for camera ################
def publish_headcount():
    success, frame = camera.read()
    if not success:
        print("Camera read failed")
        return
    
    # Use return_attention=True to get attention data
    annotated_frame, attention = detect_faces_and_pose(frame, return_attention=True)
    
    # Use ssd_net for face counting
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    ssd_net.setInput(blob)
    detections = ssd_net.forward()
    count = sum(1 for i in range(detections.shape[2]) if detections[0, 0, i, 2] >= 0.5)
    
    # Encode image
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    payload = {
        "count": count,
        "image": image_base64,
        "attentive": attention["attentive"],
        "distracted": attention["distracted"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    client.publish(TOPIC_HEADCOUNT, json.dumps(payload))
    print(f"Headcount: {count} | Attentive: {attention['attentive']} Distracted: {attention['distracted']}")


# Main loop to publish data at intervals
if __name__ == "__main__":
    last_temp_time = 0
    last_light_time = 0
    
    try:
        while True:
            now = time.time()
            
            # Temperature every 5 seconds
            if HAS_TEMP and now - last_temp_time >= 5:
                temp_result = publish_temperature()
                print(f"Temperature: {temp_result}")
                last_temp_time = now
            
            # Light every 5 seconds
            if HAS_LIGHT and now - last_light_time >= 5:
                light_result = publish_light_status()
                print(f"Light: {light_result}")
                last_light_time = now
            
            # Headcount every 1 second
            publish_headcount()
            time.sleep(1)
            
    finally:
        camera.release()
