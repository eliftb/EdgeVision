from src.camera import CameraCapture
from src.detector import YOLODetector

camera = CameraCapture()

print("YOLO yükleniyor...")
detector = YOLODetector()

print("Test yapılıyor...")
frame = camera.read()

if frame is not None:
    print(f"Frame shape: {frame.shape}")
    results = detector.detect(frame)
    print(f"Detection başarılı: {len(results[0].boxes)} box")

camera.release()