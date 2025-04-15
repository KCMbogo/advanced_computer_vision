import cv2, time, math
import numpy as np
import pulsectl
from chapters.hand_tracking_project import hand_tracking

cap = cv2.VideoCapture(0)

detector = hand_tracking.HandDetector()

min_length = 20
max_length = 300

time.sleep(2)

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = detector.find_hands(img=img)
    lmlist = detector.find_position(img=img, draw=False)
 
    if len(lmlist) != 0:  
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x2+x1)//2, (y2+y1)//2
    
        length = math.hypot((x2-x1), (y2-y1))
        
        # cv2.putText(img=img, text=f"Length: {str(int(length))}", org=(15, 30), 
        #             fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0), thickness=2, fontScale=2)
        
        normalized_length = (length - min_length) / (max_length - min_length)
        normalized_length = np.clip(normalized_length, 0.0, 1.0)

        vol_bar_height = int(250 * normalized_length) # 250 is the volume meter height
        cv2.rectangle(img=img, pt1=(50, 400), pt2=(85, 400 - vol_bar_height), color=(0, 255, 0), thickness=cv2.FILLED)
        cv2.rectangle(img=img, pt1=(50, 400), pt2=(85, 150), color=(255, 0, 0), thickness=2)
        cv2.putText(img=img, text=f"{int(normalized_length * 100)}%", org=(50, 450), color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=3, thickness=2)
        
        with pulsectl.Pulse('volume-control') as pulse:
            for sink in pulse.sink_list():
                new_volume = normalized_length
                pulse.volume_set_all_chans(sink, new_volume)

        cv2.circle(img=img, center=(x1, y1), radius=15, color=(255, 0, 255), thickness=cv2.FILLED)
        cv2.circle(img=img, center=(x2, y2), radius=15, color=(255, 0, 255), thickness=cv2.FILLED)
        cv2.circle(img=img, center=(cx, cy), radius=15, color=(255, 0, 255), thickness=cv2.FILLED)
        cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=3)
        
        if length < 50:
            cv2.circle(img=img, center=(cx, cy), radius=15, color=(0, 255, 0), thickness=cv2.FILLED)


    cv2.imshow("Frame", img)
    if cv2.waitKey(1) == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()