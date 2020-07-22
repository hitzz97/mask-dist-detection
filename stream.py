from flask import Flask, render_template, Response,redirect
from face_distance_detection.detection import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/play_stop',methods=["GET"])
def play_stop():
	change_stop()
	return "ok"


if __name__ == '__main__':
    app.run(debug=True)