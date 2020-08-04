# Mask and social distancing Detection

![Demo](relative/path/to/img.jpg?raw=true "Demo")


### Task 1 Detecting Masks
1. Detect faces using cvlib.detect_faces
2. Apply model to classifiy if the face is masked or not 

### Task 2 Detections of social distancing
3. Detect bounding boxes of person using cvlib.detect_common_objects 
4. calculate centroid for all boxes
5. feed these centroid to sklearn.DBSCAN to get the clusters.

### Multithreading 
Run both these tasks in 2 different threads.

### Display processed frames on Browser(Flask)
use streaming to stream the final frame from the app to the browser
