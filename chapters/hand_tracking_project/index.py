import cv2, time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

c_time = 0
p_time = 0

while True:
    success, img = cap.read()
    
    if not success:
        print("Failed to read the frame")
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # print(f"Multi-landmarks: {results.multi_hand_landmarks}")
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(f"Landmark: {hand_landmarks.landmark}")
            for id, landmark in enumerate(hand_landmarks.landmark):
                # print(id, landmark)
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h) # turns to pixels
                
                # cv2.circle(img=img, center=(cx, cy), radius=15, color=(0, 255, 0), thickness=cv2.FILLED, lineType=cv2.LINE_8)
                
            mp_draw.draw_landmarks(image=img, landmark_list=hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
    
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    
    cv2.putText(img=img, text=f"{str(int(fps))} fps", org=(10, 70), fontScale=1, color=(0, 255, 0), thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
