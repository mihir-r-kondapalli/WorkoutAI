import ultralytics
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8l-pose.pt')
source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')
results = model(source)  

print(results)