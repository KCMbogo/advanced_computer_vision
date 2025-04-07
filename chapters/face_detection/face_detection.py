import cv2, cvzone, time
import mediapipe as mp
from collections import namedtuple

Detection = namedtuple('Detection', ['id', 'bbox', 'score'])
Bbox = namedtuple('Bbox', ['x', 'y', 'w', 'h'])
Results = namedtuple('Results', ['img', 'detections'])

class FaceDetector:
    def __init__(self, min_detection_conf=0.5):
        self.min_detection_conf = min_detection_conf
        self.mp_face = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        
        self.face = self.mp_face.FaceDetection(min_detection_confidence=self.min_detection_conf)
        
    def detect_face(self, img):
        img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    
        results = self.face.process(img_rgb)
        
        detections = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                rbbox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = Bbox(x=int(rbbox.xmin*w), y=int(rbbox.ymin*h), w=int(rbbox.width*w), h=int(rbbox.height*h))
                score = detection.score[0]
                
                detections.append(Detection(id=id, bbox=bbox, score=score))
                
        return Results(img=img, detections=detections)
    
    def allocate_face(self, img, detections: list):
        for detection in detections:
            bbox = detection.bbox
            # cv2.rectangle(img, bbox, color=(0, 255, 0), thickness=2)
            cvzone.cornerRect(img=img, bbox=bbox, l=15, t=2, rt=3, colorC=(0, 255, 0), colorR=(0, 0, 0))
            cv2.putText(img=img, text=f'{int(detection.score*100)}%', org=(bbox.x, bbox.y-10), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)

def main():
    cap = cv2.VideoCapture("chapters/videos/face-4.mp4")
    # cap = cv2.VideoCapture(0)
    
    face_detector = FaceDetector(min_detection_conf=0.75)
    
    p_time = 0

    while True:
        success, img = cap.read()
        
        if not success:
            break
        
        img = cv2.resize(src=img, dsize=(800, 600))
        
        results = face_detector.detect_face(img=img)
        face_detector.allocate_face(img=results.img, detections=results.detections)
        
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        
        cv2.putText(img=img, text=f'{int(fps)} fps', org=(50, 50), fontScale=3, 
                    color=(0, 255, 0), thickness=2, fontFace=cv2.FONT_HERSHEY_PLAIN)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break
        
if __name__ == '__main__':
    main()