from ultralytics import YOLO

class ModelOptimizer:
    def __init__(self, model_path='yolov8n.pt'):
        # Model yükle
        self.model = YOLO(model_path)
        self.model_path = model_path

    def export_tensorrt(self):
        # TensorRT'ye dönüştür
        self.model.export(format='tensorrt')
        print("TensorRT engine oluşturuldu!")
    
    def get_engine_path(self):
        # Engine dosyasının yolunu döndür
        return self.model_path.replace('.pt', '.engine')