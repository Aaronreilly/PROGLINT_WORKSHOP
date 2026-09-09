from ultralytics import YOLO
import cv2

# ============================================================
# 1. LOAD VIDEO + MODEL
# ============================================================

model = YOLO("/home/student/proglint/runs/detect/train-9/weights/best.pt")

cap = cv2.VideoCapture("vid3.mp4")

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Video loaded")
print("Width:", width)
print("Height:", height)
print("FPS:", fps)


# ============================================================
# 2. CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "output_botsort.mp4",
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 3. FRAME BY FRAME
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ========================================================
    # 4. OBJECT DETECTION + BOT-SORT TRACKING
    # ========================================================

    results = model.track(
        frame,
        imgsz=1280,
        classes=[0],
        persist=True,

        # Use BoT-SORT instead of ByteTrack
        tracker="botsort.yaml"
    )

    result = results[0]


    # ========================================================
    # 5. GET TRACKING IDs + DRAW BOUNDING BOXES
    # ========================================================

    if result.boxes.id is not None:

        ids = result.boxes.id.int().cpu().tolist()

        boxes = result.boxes.xyxy.cpu().tolist()

        for person_id, box in zip(ids, boxes):

            x1, y1, x2, y2 = map(int, box)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Person ID
            cv2.putText(
                frame,
                f"ID: {person_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # ========================================================
    # 6. WRITE ANNOTATED FRAME
    # ========================================================

    out.write(frame)

    cv2.imshow("BoT-SORT Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()

print("Output saved as output_botsort.mp4")
