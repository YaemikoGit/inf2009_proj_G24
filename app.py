from flask import Flask, render_template, Response

#Import other microphone modules as needed

#Import other camera modules as needed
from camera.headcount import generate_frames

#Import other sensor modules as needed
from sensors.temperature import get_temperature 
from sensors.light import get_light

latest_stats = {
    "headcount": 0,
    "attentive": 0,
    "distracted": 0
}

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/stats')
def stats():
    return latest_stats

# To get light status 
@app.route('/light')
def light():
    try:
        light_status = get_light()
        return {"status": "ok", "light": light_status}
    except Exception:
        return {"status": "error", "message": "Sensor Not Detected"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
