from src.camera import CameraCapture
from src.detector import YOLODetector
from src.utils import FPSCounter
from src.recorder import VideoRecorder

# Başlat
camera = CameraCapture()
detector = YOLODetector()
fps_counter = FPSCounter()
recorder = VideoRecorder('output.mp4', 30, 640, 480)

print("30 frame işleniyor ve kaydediliyor...\n")

# 30 frame oku, detect et, kaydet
for i in range(30):
    frame = camera.read()
    
    if frame is not None:
        results = detector.detect(frame)
        boxes = len(results[0].boxes)
        fps_counter.update()
        
        recorder.write(frame)  # Frame'i kaydet
        
        print(f"Frame {i+1}: {boxes} obje | FPS: {fps_counter.get_fps():.1f}")

# Temizle
camera.release()
detector.release()
recorder.release()

print(f"\nOrtalama FPS: {fps_counter.get_fps():.1f}")
print("Program bitti!")