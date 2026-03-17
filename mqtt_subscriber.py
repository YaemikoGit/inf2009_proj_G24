import paho.mqtt.client as mqtt
import threading
import json
import ast
from flask import Flask, render_template, jsonify
from datetime import datetime
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', compression_threshold=1024)

# Store latest data separately
latest_temp = {"status": "waiting", "temperature": "No data yet", "humidity": "No data yet", "timestamp": None}
latest_light = {"status": "waiting", "light": "No data yet", "timestamp": None}
latest_headcount = {"count": 0, "image": None, "attentive": 0, "distracted": 0, "timestamp": None}
latest_sound = {"status": "waiting", "rms": None, "peak": None, "variance": None, "dBFS": None, "label": None, "timestamp": None}

# --- MQTT Setup ---
def on_message(client, userdata, message):
    global latest_temp, latest_light, latest_headcount, latest_sound
    
    try:
        raw = message.payload.decode()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = ast.literal_eval(raw)
    except Exception as e:
        print(f"Failed to parse message: {str(e)}")
        return
    
    # Add timestamp if not present
    if "timestamp" not in payload:
        payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    
    ############ for light ################
    if message.topic == "sensors/light":
        if "light" in payload and isinstance(payload["light"], bool):
            payload["light"] = "On" if payload["light"] else "Off"
        latest_light = payload
        socketio.emit("light_update", payload)
    
    
    ############ for temperature ################
    elif message.topic == "sensors/temperature":
        try:
            latest_temp = payload
            socketio.emit("temperature_update", latest_temp)
        except Exception as e:
            print(f"Failed to handle temperature message: {e}")
    
    
    ############ for camera ################
    elif message.topic == "sensors/headcount":
        # Check if this is a new image (only send if changed)
        has_new_image = payload.get("image") and payload.get("image") != latest_headcount.get("image")
        
        latest_headcount = payload
        
        # Build minimal payload
        emit_data = {
            "count": payload.get("count"),
            "timestamp": payload.get("timestamp"),
            "attentive": payload.get("attentive", 0),
            "distracted": payload.get("distracted", 0)
        }
        
        # Only send image when it actually changes
        if has_new_image:
            emit_data["image"] = payload.get("image")
        
        socketio.emit("headcount_update", emit_data)

    ############ for sound ################
    elif message.topic == "sensors/sound":
        try:
            latest_sound = payload
            socketio.emit("sound_update", latest_sound)
        except Exception as e:
            print(f"Failed to handle sound message: {e}")
            
    ############ for alerts ################
    elif message.topic.startswith("alerts/sensor_alerts/"):
        alert_type = message.topic.split("/")[-1]
        socketio.emit("sensor_alert", {"type": alert_type, "data": payload})
        print(f"Sensor Alert: {alert_type} - {payload}")
        
    elif message.topic == "alerts/attention_alerts":
        socketio.emit("attention_alert", payload)
        print(f"Attention Alert: {payload}")

mqtt_client = mqtt.Client("Subscriber")
mqtt_client.on_message = on_message
mqtt_client.connect("10.179.208.202", 1883)
mqtt_client.subscribe("sensors/temperature")
mqtt_client.subscribe("sensors/light")
mqtt_client.subscribe("sensors/headcount")
mqtt_client.subscribe("sensors/sound")
mqtt_client.subscribe("alerts/#")

# Run MQTT loop in background thread
mqtt_thread = threading.Thread(target=mqtt_client.loop_forever, daemon=True)
mqtt_thread.start()

# --- SocketIO Events ---
@socketio.on('set_attention_threshold')
def handle_set_threshold(data):
    try:
        val = float(data.get("threshold"))
        mqtt_client.publish("settings/attention_threshold", json.dumps({"threshold": val}))
        print(f"Sent new threshold to publisher: {val}")
    except Exception as e:
        print("Failed to set threshold", e)

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html',
                           light=latest_light,
                           temperature=latest_temp,
                           headcount=latest_headcount,
                           sound=latest_sound)

@app.route('/data/temperature')
def data_temperature():
    return jsonify(latest_temp)
    
@app.route('/data/light')
def data_light():
    return jsonify(latest_light)

@app.route('/data/headcount')
def data_headcount():
    return jsonify({
        "count": latest_headcount.get("count"),
        "attentive": latest_headcount.get("attentive", 0),
        "distracted": latest_headcount.get("distracted", 0),
        "timestamp": latest_headcount.get("timestamp")
    })

@app.route('/data/sound')
def data_sound():
    return jsonify(latest_sound)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)