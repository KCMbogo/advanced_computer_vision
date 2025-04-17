import cv2, time
import numpy as np
import mediapipe as mp
from chapters.pose_estimation import pose_estimation

cap = cv2.VideoCapture("projects/ai_trainer/lift2.mp4")
# cap = cv2.VideoCapture(0)

detector = pose_estimation.PoseDetector()

count = 0 # count of dumbbell curls
dir = 0 # dir = 0 going up, dir = 1 going down

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.resize(src=img, dsize=(1280, 720))
        
    # img = cv2.resize(src=cv2.imread("projects/ai_trainer/pull-up.jpg"), dsize=(640, 800))

    img = detector.detect_pose(img=img, draw=False)
    lmlist = detector.detect_position(img=img, draw=False) 
    
    if len(lmlist) != 0:
        # left arm
        angle = detector.find_angle(img=img, index_1=11, index_2=13, index_3=15)
        percent = np.interp(angle, (210, 310), (0, 100))
        gauge = np.interp(angle, (210, 310), (500, 100))
        
        color = (255, 0, 255)
        if percent == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if percent == 0:
            if dir == 1:
                count += 0.5
                dir = 0
    
    #  percent gauge
    cv2.putText(img=img, text=f"{str(int(percent))}%", org=(80, 90), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=color,
                fontScale=2, thickness=2)
    cv2.rectangle(img=img, pt1=(80, 100), pt2=(120, 500), color=color, thickness=2)
    cv2.rectangle(img=img, pt1=(80, 500), pt2=(120, int(gauge)), color=color, thickness=cv2.FILLED)

    # count box
    cv2.rectangle(img=img, pt1=(0, 600), pt2=(190, 720), color=(0, 255, 0), thickness=cv2.FILLED)    
    cv2.putText(img=img, text=str(int(count)), org=(65, 700), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                 fontScale=5, thickness=3, color=(255, 255, 255))           
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break
    
cv2.destroyAllWindows()