import cv2

class VideoRecorder:
    def __init__(self, output_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"✅ Video kaydı başladı: {output_path}")
    
    def write(self, frame):
        self.out.write(frame)
    
    def release(self):
        self.out.release()
        print("✅ Video kaydı bitti")