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
client.connect("localhost", 1883)
client.subscribe("settings/attention_threshold")
client.loop_start()

def check_and_publish_sensor_alert(topic_suffix, sensor_type, is_bad, message_detail):
    if is_bad:
        payload = {
            "sensor": sensor_type,
            "message": message_detail,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        client.publish(f"alerts/sensor_alerts/{topic_suffix}", json.dumps(payload))

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
        
        LATEST_SENSORS["temperature"] = data['temperature']
        LATEST_SENSORS["humidity"] = data['humidity']
        
        temp_val = data['temperature']
        hum_val = data['humidity']

        is_temp_bad = temp_val < 20 or temp_val > 28
        is_hum_bad = hum_val < 30 or hum_val > 70
        is_bad = is_temp_bad or is_hum_bad
        
        msg_parts = []
        if is_temp_bad:
            if temp_val < 20:
                msg_parts.append(f"Temperature too low ({temp_val}°C)")
            else:
                msg_parts.append(f"Temperature too high ({temp_val}°C)")
                
        if is_hum_bad:
            if hum_val < 30:
                msg_parts.append(f"Humidity too low ({hum_val}%)")
            else:
                msg_parts.append(f"Humidity too high ({hum_val}%)")

        msg = " | ".join(msg_parts) if msg_parts else "Normal"
        
        check_and_publish_sensor_alert("temperature_alerts", "temperature_humidity", is_bad, msg)
        
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
        
        LATEST_SENSORS["light"] = light_status
        is_light_bad = str(light_status).lower() in ["false", "off"]
        check_and_publish_sensor_alert("light_alerts", "light", is_light_bad, "Light level is off or too low")
        
    except Exception as e:
        print(f"Light sensor disconnected: {e}")
        payload = {"status": "error", "message": "Sensor Disconnected"}
    
    client.publish(TOPIC_LIGHT, json.dumps(payload))
    return payload

def calculate_blame_probabilities():
    temp_bad = LATEST_SENSORS["temperature"] < 20 or LATEST_SENSORS["temperature"] > 28
    hum_bad = LATEST_SENSORS["humidity"] < 30 or LATEST_SENSORS["humidity"] > 70
    light_bad = str(LATEST_SENSORS["light"]).lower() in ["false", "off"]
    sound_bad = LATEST_SENSORS["sound_label"].lower() in ["noisy", "loud"]
    
    bad_count = sum([temp_bad, hum_bad, light_bad, sound_bad])
    
    if bad_count > 0:
        maj_share = 0.8 / bad_count
        min_share = 0.15 / (4 - bad_count) if bad_count < 4 else 0
        others_prob = 0.05 if bad_count < 4 else 0.0
        return {
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
        light_dist = 0.1 # Base distance for OK light
        
        # Calculate sound probability based on how close the metrics are to the 'noisy' rules
        dbfs = LATEST_SENSORS.get("sound_dbfs", -50.0)
        variance = LATEST_SENSORS.get("sound_variance", 0.0)
        rms = LATEST_SENSORS.get("sound_rms", 0.0)
        peak = LATEST_SENSORS.get("sound_peak", 0.0)
        
        sound_dist = 0.0
        if variance > 0.0:
            if dbfs <= -22.92:
                if rms <= 0.03:
                    # Needs rms > 0.03 to progress towards 'noisy'. Calculate closeness.
                    sound_dist = rms / 0.03
                    if peak <= 0.38:
                        sound_dist = (sound_dist + 1.0) / 2.0  # Path align with 'noisy' peak condition
                else: 
                    # rms > 0.03, but peak > 0.38 since it wasn't alarmed as noisy
                    if peak > 0.38:
                        # Needs peak to drop <= 0.38 to become noisy
                        sound_dist = 0.38 / peak if peak > 0 else 0
            else:
                if peak <= 0.45:
                    # Needs peak > 0.45 to become noisy
                    sound_dist = peak / 0.45
                    
        # Apply base scaling for 'others' when sensors are completely perfect
        # If the environment is completely perfect (dist = 0), others will take the majority
        others_dist = 0.8
        
        total = temp_dist + hum_dist + light_dist + sound_dist + others_dist
        if total <= 0: total = 1
        
        return {
            "temperature_prob": round(temp_dist/total, 2),
            "humidity_prob": round(hum_dist/total, 2),
            "light_prob": round(light_dist/total, 2),
            "sound_prob": round(sound_dist/total, 2),
            "others_prob": round(others_dist/total, 2)
        }

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

    # Check for attention alerts
    if count > 0:
        attention_rate = attention["attentive"] / count
        if attention_rate < ATTENTION_THRESHOLD:
            probs = calculate_blame_probabilities()
            alert_payload = {
                "attention_rate": round(attention_rate, 2),
                "threshold": ATTENTION_THRESHOLD,
                "probabilities": probs,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            client.publish("alerts/attention_alerts", json.dumps(alert_payload))
            print(f"Alert: Attention dropped to {attention_rate:.2f}. Causes: {probs}")

############ for sound ################
def publish_sound():
    if not HAS_SOUND:
        return None
    try:
        payload = get_sound()
        if payload is None:
            raise RuntimeError("Sound data invalid")
        # Convert all numeric values to Python floats
        for key in ["rms", "peak", "variance", "dBFS"]:
            if key in payload:
                payload[key] = float(payload[key])
        
        # If label is a numpy type, convert to string
        if "label" in payload:
            payload["label"] = str(payload["label"])
        payload["status"] = "ok"

        LATEST_SENSORS["sound_label"] = payload.get("label", "quiet")
        LATEST_SENSORS["sound_dbfs"] = payload.get("dBFS", -50.0)
        LATEST_SENSORS["sound_variance"] = payload.get("variance", 0.0)
        LATEST_SENSORS["sound_peak"] = payload.get("peak", 0.0)
        LATEST_SENSORS["sound_rms"] = payload.get("rms", 0.0)
        
        is_sound_bad = LATEST_SENSORS["sound_label"].lower() in ["noisy", "loud"]
        check_and_publish_sensor_alert("sound_alerts", "sound", is_sound_bad, f"Noise level is high ({LATEST_SENSORS['sound_label']})")
        
    except Exception as e:
        print(f"Sound sensor error: {e}")
        payload = {"status": "error", "message": "Sound sensor not detected"}
    
    client.publish(TOPIC_SOUND, json.dumps(payload))
    return payload


# Main loop to publish data at intervals
if __name__ == "__main__":
    last_temp_time = 0
    last_light_time = 0
    last_sound_time = 0
    
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

            # Sound every 1 second
            if HAS_SOUND and now - last_sound_time >= 1:
                sound_results = publish_sound()
                print(f"Sound: {sound_results}")
                last_sound_time = now
            
            # Headcount every 1 second
            publish_headcount()
            time.sleep(1)
            
    finally:
        camera.release()
