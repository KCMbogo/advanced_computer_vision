import cv2, time
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=2):
        self.mode = mode
        self.max_hands = max_hands
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(mode, max_hands)

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:  
                if draw:                                      
                    self.mp_draw.draw_landmarks(image=img, landmark_list=hand_landmarks, connections=self.mp_hands.HAND_CONNECTIONS)

        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, landmark in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img=img, center=(cx, cy), radius=10, color=(0, 255, 0), thickness=cv2.FILLED)
                
        return landmark_list

def main():   
    cap = cv2.VideoCapture(0)
    
    detector = HandDetector()

    c_time = 0
    p_time = 0

    while True:
        success, img = cap.read()
        img = detector.find_hands(img) 
        
        if not success:
            print("Failed to read frame")
            break
        
        # img = cv2.flip(img, 1)
        
        landmark_list = detector.find_position(img=img)
        if len(landmark_list) != 0:
            print(landmark_list[4])
        
        
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
    
        cv2.putText(img=img, text=f"{str(int(fps))} fps", org=(10, 70), fontScale=1, color=(0, 255, 0), thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()