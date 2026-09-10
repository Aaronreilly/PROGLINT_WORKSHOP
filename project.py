from ultralytics import YOLO
import cv2

model = YOLO("yolo26x.pt")

seen = set()
inside = set()

in_count = 0
out_count = 0

for result in model.track(
    source="video.mp4",
    stream=True,
    imgsz=1280,
    persist=True,
    classes=[0]
):
    frame = result.plot()

    current = set()

    h, w = frame.shape[:2]

    if result.boxes.id is not None:

        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        for person_id, box in zip(ids, boxes):

            x1, y1, x2, y2 = map(int, box)
            current.add(person_id)

            # First time seeing person = IN
            if person_id not in seen:
                seen.add(person_id)
                inside.add(person_id)
                in_count += 1

            # Check if person is actually touching frame boundary
            at_edge = (
                x1 <= 5 or
                y1 <= 5 or
                x2 >= w - 5 or
                y2 >= h - 5
            )

            # Only count OUT if person reaches frame edge
            if person_id in inside and at_edge:
                inside.remove(person_id)
                out_count += 1

    # Display counts inside OpenCV frame
    cv2.putText(
        frame,
        f"IN: {in_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {out_count}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"INSIDE: {len(inside)}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Person Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()