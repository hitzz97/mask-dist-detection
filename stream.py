from flask import Flask, render_template, Response,redirect
from face_distance_detection.detection import *
import matplotlib.pyplot as plt

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

@app.route("/analytics")
def analytics():
	mask,no_mask=0,0
	mask_img=""
	with open("face.dat","r") as file:
		for line in file.readlines():
			dat=line.split()
			if len(dat)>1:
				mask+=int(dat[0])
				no_mask+=int(dat[1])
	try:
		
		mask=float(str((mask/(mask+no_mask))*100)[:4])
		no_mask=float(str((no_mask/(mask+no_mask))*100)[:4])

		print(mask,no_mask)
		labels = 'Mask Detected', 'No Mask Detected'
		sizes = [mask, no_mask]
		explode = (0, 0.1)  

		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
		        shadow=True, startangle=90)
		ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
		plt.savefig("static/mask.jpg")
		mask_img="mask.jpg"
	except Exception as e:
		print(e)

	total_person,violators = 0,0
	follow=[]
	violate=[]
	ana_img=""
	overTime=""
	with open("person.dat","r") as file:
		for line in file.readlines():
			dat=line.split()
			if len(dat)>1:
				total_person+=int(dat[0])
				follow.append(int(dat[0])-int(dat[1]))
				violators+=int(dat[1])
				violate.append(int(dat[1]))

	followers = total_person-violators
	try:
		followers=float(str((followers/(total_person))*100)[:4])
		violators=float(str((violators/(total_person))*100)[:4])
		print(followers,violators)

		labels = 'followers', 'violators'
		sizes = [followers, violators]
		explode = (0, 0.1)  

		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
		        shadow=True, startangle=90)
		ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
		plt.savefig("static/analysis.jpg")
		ana_img="analysis.jpg"

		x=[i for i in range(len(follow))]
		plt.figure(figsize=(8,4),dpi=80,facecolor="w",edgecolor="k")
		plt.plot(x,follow,label="followers",linewidth=3)
		plt.plot(x,violate,label="violaters",linewidth=3)
		plt.legend(['follower',"violaters"])
		plt.xlabel("Time")
		plt.ylabel("Count")
		plt.savefig("static/overTime.jpg")
		overTime="overTime.jpg"
	except Exception as e:
		print(e)

	return render_template("analysis.html",mask=mask,no_mask=no_mask,followers=followers,
			violators=violators,mask_img=mask_img,ana_img=ana_img,overTime=overTime
			)

if __name__ == '__main__':
    app.run(debug=True)