import cv2, time
import mediapipe as mp

cap = cv2.VideoCapture("chapters/videos/pose-1.mp4")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

p_time = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.resize(img, (800, 600))
    img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        mp_draw.draw_landmarks(image=img, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(landmark.x*w), int(landmark.y*h)
            cv2.circle(img=img, center=(cx, cy), radius=5, color=(0, 255, 0), thickness=cv2.FILLED)
    
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    
    cv2.putText(img=img, text=f"{str(int(fps))} fps", org=(50, 50), fontScale=2, color=(255, 0, 0), 
                thickness=3, fontFace=cv2.FONT_HERSHEY_PLAIN)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break