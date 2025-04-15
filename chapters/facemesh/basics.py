import cv2, time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_facemesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

facemesh = mp_facemesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

p_time = 0

while True:
    success, img = cap.read()
    
    if not success:
        break

    img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    
    results = facemesh.process(img_rgb)
    # print(results.multi_face_landmarks)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(image=img, landmark_list=face_landmarks, connections=mp_facemesh.FACEMESH_CONTOURS, 
                                   landmark_drawing_spec=draw_spec)
            
            for id, landmark in enumerate(face_landmarks.landmark):
                h, w, c = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                # print(id, x, y)
        
    
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    
    cv2.putText(img=img, text=str(int(fps)), org=(50, 50), color=(0, 255, 0), thickness=2, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()