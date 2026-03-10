import paho.mqtt.client as mqtt
import threading
import json
import ast
from flask import Flask, render_template, jsonify
from datetime import datetime
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store latest data separately
latest_temp = {"status": "waiting", "temperature": "No data yet", "humidity": "No data yet", "timestamp": None}
latest_light = {"status": "waiting", "light": "No data yet", "timestamp": None}
latest_headcount = {"count": 0, "image": None, "attentive": 0, "distracted": 0, "timestamp": None}

# --- MQTT Setup ---
def on_message(client, userdata, message):
    global latest_temp, latest_light, latest_headcount
    
    try:
        raw = message.payload.decode()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = ast.literal_eval(raw)
    except Exception as e:
        print(f"Failed to parse message: {str(e)}")
        return
    
    payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if message.topic == "sensors/light":
        if "light" in payload and isinstance(payload["light"], bool):
            payload["light"] = "On" if payload["light"] else "Off"
        latest_light = payload
        print(f"Light update: {payload}")
        socketio.emit("light_update", payload)
        
    elif message.topic == "sensors/temperature":
        try:
            latest_temp = payload
            latest_temp["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Temperature update: {latest_temp}")
            socketio.emit("temperature_update", latest_temp)
        except Exception as e:
            print(f"Failed to handle temperature message: {e}")
    
    elif message.topic == "sensors/headcount":
        latest_headcount = payload
        print(f"Headcount: {payload.get('count')} | Attentive: {payload.get('attentive', 0)} Distracted: {payload.get('distracted', 0)}")
        socketio.emit("headcount_update", {
            "count": payload.get("count"),
            "timestamp": payload.get("timestamp"),
            "image": payload.get("image"),
            "attentive": payload.get("attentive", 0),
            "distracted": payload.get("distracted", 0)
        })

mqtt_client = mqtt.Client("Subscriber")
mqtt_client.on_message = on_message
mqtt_client.connect("10.48.179.202", 1883)
mqtt_client.subscribe("sensors/temperature")
mqtt_client.subscribe("sensors/light")
mqtt_client.subscribe("sensors/headcount")

# Run MQTT loop in background thread
mqtt_thread = threading.Thread(target=mqtt_client.loop_forever, daemon=True)
mqtt_thread.start()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html',
                           light=latest_light,
                           headcount=latest_headcount)

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

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)