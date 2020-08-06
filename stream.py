from flask import Flask, render_template, Response,redirect
from face_distance_detection.detection import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<times>')
def video_feed(times):
    return Response(detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/play_stop',methods=["GET"])
def play_stop():
	change_stop()
	return "ok"

@app.route('/change_input/<num>',methods=["GET"])
def change_input(num):
	change_input_file(int(num))
	return "ok"

@app.route('/change_FPS/<num>',methods=["GET"])
def change_fps(num):
	change_FPS(int(num))
	return "ok"

@app.route('/info')
def info():
	# person_cnt,no_mask,dist_vio
	return "#".join([str(i) for i in get_info()]) 

if __name__ == '__main__':
    app.run(debug=True)