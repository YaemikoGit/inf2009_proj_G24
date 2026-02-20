from flask import Flask, render_template, Response

#Import other microphone modules as needed


#Import other camera modules as needed
from camera.headcount import generate_frames, detect_faces_and_pose

#Import other sensor modules as needed
from sensors.temperature import get_temperature 

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)