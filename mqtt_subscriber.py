import paho.mqtt.client as mqtt
import threading
import json
import ast
import os
import csv
import time
import glob
from flask import Flask, render_template, jsonify, send_from_directory, request
from datetime import datetime
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', compression_threshold=1024)

# Logging Setup
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
is_recording = False
record_thread = None
current_attention_threshold = 0.50

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
        print(f"Light update: {payload}")
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
        latest_headcount = payload

        if payload.get("status") == "error":
            # handle error message 
            socketio.emit("headcount_error", payload)
        else:
            # Always send all data including image
            emit_data = {
                "count": payload.get("count"),
                "timestamp": payload.get("timestamp"),
                "attentive": payload.get("attentive", 0),
                "distracted": payload.get("distracted", 0),
                "image": payload.get("image")  # Always include, even if None
            }
        
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
mqtt_client.connect("172.20.10.2", 1883) # CHANGE IP TO YOUR PI'S IP ADDRESS
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
    global current_attention_threshold
    try:
        val = float(data.get("threshold"))
        current_attention_threshold = val
        mqtt_client.publish("settings/attention_threshold", json.dumps({"threshold": val}))
        print(f"Sent new threshold to publisher: {val}")
    except Exception as e:
        print("Failed to set threshold", e)

def manage_log_files():
    """Ensure only the 10 most recent log files are kept."""
    try:
        files = glob.glob(os.path.join(LOGS_DIR, "session_*.csv"))
        files.sort(key=os.path.getctime)
        while len(files) > 10:
            os.remove(files.pop(0))
    except Exception as e:
        print(f"Error managing log files: {e}")

def recording_loop():
    global is_recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.csv"
    filepath = os.path.join(LOGS_DIR, filename)
    
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Attentive", "Distracted", "Attention_Rate", "Attention_Threshold",
            "Light", "Temperature", "Humidity", 
            "Sound_dBFS", "Sound_RMS", "Sound_Peak", "Sound_Variance"
        ])
    
    manage_log_files()
    print(f"Started recording to {filename}")

    while is_recording:
        try:
            # Headcount
            hc = latest_headcount
            if hc.get("status") == "error" or hc.get("count", 0) == 0:
                attentive, distracted, att_rate = "NaN", "NaN", "NaN"
            else:
                attentive = hc.get("attentive", 0)
                distracted = hc.get("distracted", 0)
                total = attentive + distracted
                att_rate = round(attentive / total, 2) if total > 0 else 0.0

            # Light
            lt = latest_light
            if lt.get("status") == "error" or lt.get("light") in ["No data yet", None]:
                light_val = "NaN"
            else:
                light_str = str(lt.get("light", "")).lower()
                if light_str in ["on", "true", "1"]: light_val = 1
                elif light_str in ["off", "false", "0"]: light_val = 0
                else: light_val = "NaN"

            # Temp/Hum
            tm = latest_temp
            if tm.get("status") == "error" or tm.get("temperature") == "No data yet":
                temp_val, hum_val = "NaN", "NaN"
            else:
                temp_val = tm.get("temperature", "NaN")
                hum_val = tm.get("humidity", "NaN")

            # Sound
            sd = latest_sound
            if sd.get("status") == "error" or sd.get("dBFS") is None:
                snd_dbfs, snd_rms, snd_peak, snd_var = "NaN", "NaN", "NaN", "NaN"
            else:
                snd_dbfs = sd.get("dBFS", "NaN")
                snd_rms = sd.get("rms", "NaN")
                snd_peak = sd.get("peak", "NaN")
                snd_var = sd.get("variance", "NaN")

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(filepath, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    now_str, attentive, distracted, att_rate, current_attention_threshold,
                    light_val, temp_val, hum_val,
                    snd_dbfs, snd_rms, snd_peak, snd_var
                ])
        except Exception as e:
            print(f"Error logging data: {e}")

        # Sleep ~1s, but check flag often so we can stop quickly
        for _ in range(10):
            if not is_recording:
                break
            time.sleep(0.1)

@socketio.on('toggle_recording')
def handle_toggle_recording(data):
    global is_recording, record_thread
    action = data.get("action")
    if action == "start" and not is_recording:
        is_recording = True
        record_thread = threading.Thread(target=recording_loop, daemon=True)
        record_thread.start()
        socketio.emit("recording_status", {"status": "recording"})
    elif action == "stop" and is_recording:
        is_recording = False
        socketio.emit("recording_status", {"status": "stopped"})

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

@app.route('/logs')
def view_logs():
    try:
        files = glob.glob(os.path.join(LOGS_DIR, "session_*.csv"))
        files.sort(key=os.path.getctime, reverse=True)
        filenames = [os.path.basename(f) for f in files]
    except Exception:
        filenames = []
    return render_template('logs.html', log_files=filenames)

@app.route('/api/logs/<filename>', methods=['GET'])
def get_log_file(filename):
    if filename.endswith('.csv') and '..' not in filename:
        return send_from_directory(os.path.abspath(LOGS_DIR), filename)
    return "Invalid file", 400

@app.route('/api/logs/download/<filename>', methods=['GET'])
def download_log_file(filename):
    if filename.endswith('.csv') and '..' not in filename:
        return send_from_directory(os.path.abspath(LOGS_DIR), filename, as_attachment=True)
    return "Invalid file", 400

@app.route('/api/logs/<filename>', methods=['DELETE'])
def delete_log_file(filename):
    if filename.endswith('.csv') and '..' not in filename:
        filepath = os.path.join(LOGS_DIR, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Invalid file"}), 400

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)