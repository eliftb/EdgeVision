from src.optimizer import ModelOptimizer

# Optimizer oluştur
optimizer = ModelOptimizer('yolov8n.pt')

# TensorRT'ye çevir (İLK KERE İNDİRİLECEK!)
print("TensorRT'ye dönüştürülüyor...")
optimizer.export_tensorrt()

# Engine path'i al
engine_path = optimizer.get_engine_path()
print(f"Engine path: {engine_path}")
