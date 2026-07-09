# EdgeVision

Edge Device Üzerinde Gerçek Zamanlı Bilgisayarlı Görü Pipeline'ı

## Özellikler

- 📹 Gerçek zamanlı kamera capture (640x480 @ 30 FPS)
- 🤖 YOLOv8 object detection (GPU accelerated)
- 📊 FPS counter ve performance monitoring
- 🎬 Video recording (MP4 format)

## Performans

- Average FPS: 6.8
- Inference Time: 10.7ms
- GPU: NVIDIA RTX 4060
- Model: YOLOv8n (nano)

## Proje Yapısı
src/
├── camera.py (Kamera pipeline)
├── detector.py (YOLO detection)
├── recorder.py (Video kayıt)
└── utils.py (FPS counter)
main.py (Ana program)

## Kullanım

```bash
python3 main.py
```

## Sınav Hazırlığı

Bu proje Baykar "Edge Device Üzerinde Uçtan Uca Görü Modellerinin Çalıştırılması" 
sınavı için hazırlanmıştır.
