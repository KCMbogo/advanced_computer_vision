import cv2, time
import mediapipe as mp
from collections import namedtuple

class FaceMesh:
    def __init__(self, max_num_faces, thickness=1, radius=2, color=(0, 255, 0)):
        self.max_num_faces = max_num_faces
        self.mp_facemesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        self.facemesh = self.mp_facemesh.FaceMesh(max_num_faces=2)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=thickness, circle_radius=radius, color=color)
        
    def face_mesh(self, img, draw: bool=False):
        img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        results = self.facemesh.process(img_rgb)
        
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw: 
                    self.mp_draw.draw_landmarks(image=img, landmark_list=face_landmarks, 
                                                connections=self.mp_facemesh.FACEMESH_CONTOURS, 
                                                landmark_drawing_spec=self.draw_spec)
                face = []
                for id, landmark in enumerate(face_landmarks.landmark):
                    h, w, c = img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    face.append([x, y])
                faces.append(face)
        return img, faces
                    
                    
def main():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("chapters/videos/face-4.mp4")
    face_mesh = FaceMesh(max_num_faces=2)
    
    p_time = 0
    while True:
        success, img = cap.read()
        
        if not success:
            break
        
        img = cv2.resize(src=img, dsize=(800, 600))
        
        img, faces = face_mesh.face_mesh(img=img, draw=True)
        
        # if len(faces) != 0:
        #     print(len(faces))
        
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        
        cv2.putText(img=img, text=str(int(fps)), org=(50, 50), color=(0, 255, 0), thickness=2, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()