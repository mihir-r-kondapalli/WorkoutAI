import ultralytics
from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8l-pose.pt')
vid = cv2.VideoCapture(0) 

if not vid.isOpened():
    print("Error: Could not open camera.")
    exit()



