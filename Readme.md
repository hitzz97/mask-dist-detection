# Mask and social distancing Detection

### Task 1 Detecting Masks
1. Detect faces using cvlib.detect_faces
2. Apply model to classifiy if the face is masked or not 

### Task 2 Detections of social distancing
3. Detect bounding boxes of person using cvlib.detect_common_objects 
4. calculate centroid for all boxes
5. feed these centroid to sklearn.DBSCAN to get the clusters.

<img src="https://github.com/hitzz97/mask-dist-detection/blob/master/demo1.jpg" height="350" width="800"/>

### Multithreading 
Run both these tasks in 2 different threads.

### Display processed frames on Browser(Flask)
use streaming to stream the final frame from the app to the browser

### Realtime and Post Session Statistics of Detections 
Realtime person, distance violators and un masked count display
Post Session statistics of overall detection with time for analysis.

<img src="https://github.com/hitzz97/mask-dist-detection/blob/master/demo2.jpg" height="350" width="800"/>