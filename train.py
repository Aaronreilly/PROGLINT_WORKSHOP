from ultralytics import YOLO

# Load pretrained YOLOv8 small
model = YOLO("yolov8s.pt")

# Train
model.train(
    data="human/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    classes=[0]
)
