import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.model.to('cuda')
        print("YOLO modeli yüklendi (GPU)")
    
    def detect(self, frame):
        results = self.model(frame, device=0)
        return results
    
    def release(self):
        self.model = None
        print("YOLO modeli temizlendi")