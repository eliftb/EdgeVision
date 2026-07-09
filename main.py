from src.camera import CameraCapture
from src.detector import YOLODetector
from src.utils import FPSCounter

# Başlat
camera = CameraCapture()
detector = YOLODetector()
fps_counter = FPSCounter()

print("30 frame işleniyor...\n")

# 30 frame oku, detect et, FPS hesapla
for i in range(30):
    frame = camera.read()
    
    if frame is not None:
        results = detector.detect(frame)
        boxes = len(results[0].boxes)
        fps_counter.update()
        
        print(f"Frame {i+1}: {boxes} obje | FPS: {fps_counter.get_fps():.1f}")

# Temizle
camera.release()
detector.release()

print(f"\nOrtalama FPS: {fps_counter.get_fps():.1f}")
print("Program bitti!")