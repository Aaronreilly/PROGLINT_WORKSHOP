from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture("vid3.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        imgsz=640,
        conf=0.5,
        classes=[0],
        verbose=False
    )

    frame = results[0].plot()

    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
