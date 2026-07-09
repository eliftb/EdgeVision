import cv2

class CameraCapture:
    def __init__(self, camera_id=0):    
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("X Kamera açılamadı!")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"✅ Kamera açıldı: {self.width}x{self.height} @ {self.fps} FPS")

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            return None
    
        return frame

    def release(self):
        self.cap.release()
        print("✅ Kamera kapalı")