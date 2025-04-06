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
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:  
                if draw:                                      
                    self.mp_draw.draw_landmarks(image=img, landmark_list=hand_landmarks, connections=self.mp_hands.HAND_CONNECTIONS)

        return img

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