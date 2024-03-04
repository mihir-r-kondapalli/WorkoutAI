from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8n-pose.pt') 
vid = cv2.VideoCapture(0) 

while(True):
    ret, frame = vid.read()
    start_time = time.time()
    # Run YOLOv8 inference on the frame
    results = model(frame, verbose = False)

        
    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # print keypoints index number and x,y coordinates
    keypoints = results[0].keypoints[0].xy.cpu().numpy()
    print(keypoints)
    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    cv2.putText(annotated_frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break