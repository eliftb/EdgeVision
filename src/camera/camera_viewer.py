import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera Açılmadı")
    exit()

printed = False

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Frame okunamadı!")
        continue
    
    # İlk frame'de bilgileri yazdır
    if not printed:
        height, width, channels = frame.shape
        print(f"Height : {height}")
        print(f"Width  : {width}")
        print(f"Channels : {channels}") 
        printed = True
    
    # Frame'i göster
    cv2.imshow("Camera", frame)
    
    # q tuşuna basılırsa kapat
    if cv2.waitKey(1) == ord('q'):
        print("Program kapatılıyor...")
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()  # ✅ Parantez eklendi

print("Program bitti!")