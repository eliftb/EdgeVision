import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera Açılmadı")
    exit()
printed = False

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    if not printed:
        height, width, channels = frame.shape

        print(f"Height : {height}")
        print(f"Width  : {width}")
        print(f"Channels : {channels}") 
        printed = True

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

print(frame.shape)

cv2.destroyAllWindows
