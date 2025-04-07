import cv2, time
import mediapipe as mp


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()

    def detect_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(image=img, landmark_list=self.results.pose_landmarks, connections=self.mp_pose.POSE_CONNECTIONS)
            
        return img
    
    def detect_position(self, img, draw=True):
        landmark_list = []
        for id, landmark in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(landmark.x*w), int(landmark.y*h)
            landmark_list.append([id, cx, cy])
            
            if draw:
                cv2.circle(img=img, center=(cx, cy), radius=5, color=(0, 255, 0), thickness=cv2.FILLED)

        return landmark_list
    
    
def main():
    cap = cv2.VideoCapture("chapters/videos/pose-6.mp4")

    p_time = 0
    
    pose = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.resize(img, (800, 600))
        
        img = pose.detect_pose(img=img)
        
        landmark_list = pose.detect_position(img=img)
        
        if len(landmark_list) != 0:
            print(landmark_list[4])
        
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        
        cv2.putText(img=img, text=f"{str(int(fps))} fps", org=(50, 50), fontScale=2, color=(255, 0, 0), 
                    thickness=3, fontFace=cv2.FONT_HERSHEY_PLAIN)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("q"):
            break
        
        
if __name__ == "__main__":
    main()