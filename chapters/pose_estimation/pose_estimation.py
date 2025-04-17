import cv2, time, math
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
        self.landmark_list = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                self.landmark_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img=img, center=(cx, cy), radius=2, color=(0, 255, 0), thickness=cv2.FILLED)

        return self.landmark_list
    
    def find_angle(self, img, index_1, index_2, index_3, draw=True):
        x1, y1 = self.landmark_list[index_1][1:]
        x2, y2 = self.landmark_list[index_2][1:]
        x3, y3 = self.landmark_list[index_3][1:]
        
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        
        if angle < 0:
            angle += 360
        
        if draw:
            cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=3)
            cv2.line(img=img, pt1=(x3, y3), pt2=(x2, y2), color=(255, 255, 255), thickness=3)
            cv2.circle(img=img, center=(x1, y1), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(img=img, center=(x1, y1), radius=15, color=(0, 0, 255), thickness=2)
            cv2.circle(img=img, center=(x2, y2), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(img=img, center=(x2, y2), radius=15, color=(0, 0, 255), thickness=2)
            cv2.circle(img=img, center=(x3, y3), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)
            cv2.circle(img=img, center=(x3, y3), radius=15, color=(0, 0, 255), thickness=2)
            # cv2.putText(img=img, text=str(int(angle)), org=(x2+10, y2+30), 
            #             fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2)
        return angle
    
    
def main():
    cap = cv2.VideoCapture("chapters/videos/pose-6.mp4")
    # cap = cv2.VideoCapture(0)
    
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