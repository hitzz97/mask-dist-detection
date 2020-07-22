import time
print("\n[STAT] Starting Imports\n\n")
calc=time.time()
import cv2
import cvlib as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from threading import Thread
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier as KNNC 
import pickle
import gc
import face_recognition
import PIL.Image as Image
import io,gc

#IMPORTS
print ("\n\n[STAT] import completed in %ssec. \n" %(round(time.time()-calc)))

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
frame                  = None
faces_list             = []
obj_list               = []
stop                   = True
#classes = ["mask_weared_incorrect","with_mask","without_mask"]
classes = ["with mask","no mask"]
t=time.time()
fps=0
counter=0
d=(640,480)
threaded=0
threshold_distance=75

model=models.load_model("85%.h5")
print("\n[STAT] Model Loaded\n")


#FACE DETECTION THREAD
def face_det():
    global faces_list
    while not stop:

        try:
            var_frame=frame[:]
            faces, confidences = cv.detect_face(var_frame,enable_gpu=False,threshold=0.2)

            ls=[]
            for box,conf in zip(faces,confidences):
                crop_face = var_frame[box[1]:box[3], box[0]:box[2]]

                crop_face=cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
                rs_face=cv2.resize(crop_face,(256,256), interpolation = cv2.INTER_AREA)
                
                p=model.predict(rs_face.reshape(1,256,256,3))
                p = np.argmax(p, axis=1)
                #p=knn.predict(p)
                p=classes[int(p[0])]

                if p=="with mask":
                    color = (0, 255, 0)
                else :
                    color = (0, 0, 255)

                ls.append([box,p,color])
        except:

            ls=[]

        faces_list=ls[:]
        # print(faces_list)

    return

#OBJECT Detection Thread 
def obj_det():
    
    global obj_list
    
    while not stop:
        
        resz_frame=cv2.resize(frame,(160,120))
        with tf.device("/CPU:0"):
            b,l,c = cv.detect_common_objects(resz_frame,enable_gpu=False,model='yolov3lite')
            ls1=[]
            for box,label,conf in zip(b,l,c):
                if label=='person':
                    x1=int(box[0]*d[0]/160)
                    y1=int(box[1]*d[1]/120)
                    x2=int(box[2]*d[0]/160)
                    y2=int(box[3]*d[1]/120)

                    x=int(x1+(x2-x1)/2)
                    y=int(y2+(y1-y2)/2)

                    color = list(np.random.random(size=3) * 256)
                    ls1.append((x,y))

            obj_list=ls1[:]
    return

FPS=10
def track_FPS():
    global fps,counter,t,FPS

    prev=time.time()
    while(time.time()-prev<(1/FPS)):
        pass
        
    if time.time()-t>=1:
        fps=counter
        t=time.time()
        counter=0
    
    else:
        counter+=1
    return

def change_stop():
    global stop
    stop=not stop



def detection():
    global frame,t1,t2,faces_list,obj_list,threaded,stop
    print("detection\n\n")
    t1=Thread(target=face_det)
    t2=Thread(target=obj_det)
    webcam = cv2.VideoCapture(0)
    # webcam = cv2.VideoCapture("t.mp4")

    #knn = pickle.load(open('knn_clas.pkl', 'rb'))

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    else: #get dimension info

        width =  int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dim = (width, height)
        print('Original Dimensions : ',dim)
        
    while webcam.isOpened() and not stop:    
        # read frame from webcam 
        status, raw_frame = webcam.read()
        frame=cv2.resize(raw_frame,d)
        
        if not status:
            break

        if not threaded:
            threaded=1
            t1.start()
            t2.start()
            
        track_FPS()
        
    #     print(faces_list)
        for item in faces_list:
            box=item[0]
            p=item[1]
            color=item[2]
            

            start_point = (box[0],box[1])
            end_point = (box[2], box[3])
            thickness = 2
            cv2.rectangle(frame, start_point, end_point, color, thickness)

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (box[0],box[1]-5)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2

            cv2.putText(frame,p, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
    #     print(obj_list)

        try:
            clustering = DBSCAN(eps=threshold_distance,min_samples=2).fit(obj_list)
            isSafe = clustering.labels_

            arg_sorted = np.argsort(isSafe)
            for i in range(1,len(obj_list)):

                if isSafe[arg_sorted[i]]!=-1 and isSafe[arg_sorted[i]]==isSafe[arg_sorted[i-1]]:
                    cv2.line(frame,obj_list[arg_sorted[i]],obj_list[arg_sorted[i-1]],(0,0,255),2)

            # Put Bounding Boxes on People in the Frame
            for p in range(len(obj_list)):

                x,y = obj_list[p]

              # Green if Safe, Red if UnSafe
                if isSafe[p]==-1:
                    cv2.circle(frame, (x,y), radius=10, color=(0, 255, 255), thickness=-1)
                else:
                    cv2.circle(frame, (x,y), radius=10, color=(0, 0, 255), thickness=-1)
        except:
            pass
              
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,255,255)

        cv2.putText(frame,"FPS:"+str(fps),(40,50), 
            font, 
            fontScale,
            fontColor,
            )
    #     frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Person detection", frame)
        pil_im = Image.fromarray(frame)
        b = io.BytesIO()
        pil_im.save(b, 'jpeg')
        im_bytes = b.getvalue()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + im_bytes + b'\r\n') 

        # key = cv2.waitKey(1) & 0xff

        # if key == ord('q'):
        #     stop=True
        #     t1.join()
        #     t2.join()
        #     break


    # release resources
    t1.join()
    t2.join()
    webcam.release()
    gc.collect()
    # cv2.destroyAllWindows()
# detection()