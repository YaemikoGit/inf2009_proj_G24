# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import time
from datetime import datetime
import json
import base64
import cv2
from sensors.light import get_light
from camera.headcount import detect_faces_dnn, net, camera, detect_faces_and_pose

# MQTT Setup
client = mqtt.Client("Publisher")
client.connect("192.168.137.42", 1883)

TOPIC_LIGHT = "sensors/light"
TOPIC_HEADCOUNT = "sensors/headcount"

last_headcount = -1

def publish_light_status():
    try:
        light_status = get_light()
        payload = {"status": "ok", "light": light_status}
    except Exception:
        payload = {"status": "error", "message": "Sensor Not Detected"}
    client.publish(TOPIC_LIGHT, json.dumps(payload))
    return payload

# def publish_headcount():
#     global last_headcount
#     success, frame = camera.read()  # uses headcount.py's camera instance
#     if not success:
#         print("Camera read failed")
#         return

#     annotated_frame = detect_faces_dnn(frame)

#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#     count = sum(1 for i in range(detections.shape[2]) if detections[0, 0, i, 2] >= 0.5)

#     if count != last_headcount:
#         last_headcount = count
#         _, buffer = cv2.imencode('.jpg', annotated_frame)
#         image_base64 = base64.b64encode(buffer).decode('utf-8')
#         payload = {"count": count, "image": image_base64}
#         client.publish(TOPIC_HEADCOUNT, json.dumps(payload))
#         print(f"Headcount changed: {count} - published with image")
#     else:
#         print(f"Headcount unchanged: {count} - skipped")


TOPIC_HEADCOUNT = "sensors/headcount"

def publish_headcount():
    success, frame = camera.read()
    if not success:
        print("Camera error")
        return

    headcount, results = detect_faces_and_pose(frame)

    attentive = 0
    distracted = 0

    for r in results:
        yaw = r["yaw"]
        pitch = r["pitch"]

        if abs(yaw) < 25 and abs(pitch) < 20:
            attentive += 1
        else:
            distracted += 1

    # Take a snapshot ONLY when sending
    _, buffer = cv2.imencode('.jpg', frame)
    snapshot = base64.b64encode(buffer).decode("utf-8")

    payload = {
        "count": headcount,
        "attentive": attentive,
        "distracted": distracted,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot": snapshot
    }

    client.publish(TOPIC_HEADCOUNT, json.dumps(payload))
    print("MQTT sent:", payload)


if __name__ == "__main__":
    last_light_time = 0
    try:
        while True:
            now = time.time()

            # Light every 5 seconds
            if now - last_light_time >= 5:
                light_result = publish_light_status()
                print(f"Light: {light_result}")
                last_light_time = now

            # Headcount every 1 second
            publish_headcount()

            time.sleep(1)
    finally:
        camera.release()
