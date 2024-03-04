import ultralytics
from ultralytics import YOLO
import cv2
import time
import os

model = YOLO('yolov8l-pose.pt')
# access our camera
vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()
    results = model(frame, verbose = False)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 frame', annotated_frame)
    cv2.putText(annotated_frame, 'OpenCV', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, 34)
    # press q to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
    #need put text

#if not vid.isOpened():
    #print("Error: Could not open camera.")
    #exit()