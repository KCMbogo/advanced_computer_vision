import cv2, time
import mediapipe as mp

cap = cv2.VideoCapture("chapters/videos/face-4.mp4")

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face = mp_face.FaceDetection(min_detection_confidence=0.75)

p_time = 0

while True:
    success, img = cap.read()
    
    if not success:
        break
    
    img = cv2.resize(src=img, dsize=(800, 600))
    img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    
    results = face.process(img_rgb)
    # print(results.detections)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(f'Score: {detection.score} and Location data: {detection.location_data}')
            # print(f"KEY POINTS: {detection.location_data.relative_keypoints}")
            # mp_draw.draw_detection(image=img, detection=detection)
            
            bbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            x1, y1 = int(bbox.xmin*w), int(bbox.ymin*h)
            width, height = int(bbox.width*w), int(bbox.height*h)
            score = detection.score[0]
            
            cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x1+width, y1+height), color=(0, 255, 0), thickness=2)
            cv2.putText(img=img, text=f'{int(score*100)}%', org=(x1, y1-10), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)

    
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    
    cv2.putText(img=img, text=f'{int(fps)} fps', org=(50, 50), fontScale=3, 
                color=(0, 255, 0), thickness=3, fontFace=cv2.FONT_HERSHEY_PLAIN)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break