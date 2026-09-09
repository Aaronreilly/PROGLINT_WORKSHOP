from ultralytics import YOLO

import cv2
import os


# ============================================================
# 1. PATHS
# ============================================================

VIDEO_PATH = "vid3.mp4"

FRAMES_DIR = "frames"
ANNOTATED_FRAMES_DIR = "annotated_frames"
TRACKED_FRAMES_DIR = "tracked_frames"

OUTPUT_VIDEO = "output_botsort.mp4"

MODEL_PATH = "/home/student/proglint/runs/detect/train-9/weights/best.pt"


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)
os.makedirs(TRACKED_FRAMES_DIR, exist_ok=True)


# ============================================================
# 3. LOAD YOLO MODEL
# ============================================================

print("======================================")
print("LOADING YOLO MODEL")
print("======================================")

model = YOLO(MODEL_PATH)

print("YOLO model loaded")
print()


# ============================================================
# 4. LOAD VIDEO
# ============================================================

print("======================================")
print("LOADING VIDEO")
print("======================================")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0


print("Video loaded")
print("Width :", width)
print("Height:", height)
print("FPS   :", fps)
print()


# ============================================================
# 5. EXTRACT VIDEO FRAMES
# ============================================================

print("======================================")
print("STEP 1: EXTRACTING FRAMES")
print("======================================")

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_path = os.path.join(
        FRAMES_DIR,
        f"frame_{frame_count:06d}.jpg"
    )

    cv2.imwrite(
        frame_path,
        frame
    )

    frame_count += 1


cap.release()

print("Frames extracted:", frame_count)
print()


# ============================================================
# 6. BOT-SORT CONFIGURATION
# ============================================================

print("======================================")
print("BOT-SORT CONFIGURATION")
print("======================================")

# IMPORTANT:
#
# Do NOT create BOTSORT manually.
#
# Do NOT do:
#
#     tracker = BOTSORT(...)
#
# Do NOT call:
#
#     tracker.update(...)
#
# Ultralytics internally creates and manages BoT-SORT when:
#
#     tracker="botsort.yaml"
#
# is passed to model.track().
#
# This avoids the BOTSORT API mismatch in your version.

TRACKER_CONFIG = "botsort.yaml"

print("Tracker:", TRACKER_CONFIG)
print("BoT-SORT will be managed by Ultralytics.")
print()


# ============================================================
# 7. PROCESS EACH FRAME
# ============================================================

print("======================================")
print("STEP 2: YOLO + BOT-SORT TRACKING")
print("======================================")

processed_frame_count = 0

for frame_number in range(frame_count):

    # ========================================================
    # LOAD FRAME
    # ========================================================

    frame_path = os.path.join(
        FRAMES_DIR,
        f"frame_{frame_number:06d}.jpg"
    )

    frame = cv2.imread(frame_path)

    if frame is None:

        print(
            "Could not read:",
            frame_path
        )

        continue


    # ========================================================
    # YOLO + BOT-SORT
    # ========================================================
    #
    # persist=True is VERY IMPORTANT.
    #
    # It tells Ultralytics that this frame is the next frame
    # of the same video sequence.
    #
    # Therefore BoT-SORT can maintain IDs between frames.
    #
    # tracker="botsort.yaml" selects BoT-SORT.
    #
    # classes=[0] means only class 0.
    #
    # imgsz=1280 controls YOLO inference resolution.
    #
    # conf=0.25 is the detection confidence threshold.
    #
    # verbose=False prevents YOLO from printing information
    # for every frame.
    # ========================================================

    results = model.track(
        source=frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        imgsz=1280,
        classes=[0],
        conf=0.25,
        verbose=False
    )


    result = results[0]


    # ========================================================
    # CREATE OUTPUT FRAME
    # ========================================================

    annotated_frame = frame.copy()


    # ========================================================
    # GET TRACKING DATA
    # ========================================================

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()

        confidences = result.boxes.conf.cpu().numpy()

        classes = result.boxes.cls.cpu().numpy()


        # ----------------------------------------------------
        # GET TRACK IDs
        # ----------------------------------------------------

        if result.boxes.id is not None:

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .numpy()
            )

        else:

            track_ids = [None] * len(boxes)


        # ====================================================
        # DRAW EACH TRACK
        # ====================================================

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box

            confidence = float(confidences[i])

            class_id = int(classes[i])


            # ------------------------------------------------
            # Convert coordinates to integers
            # ------------------------------------------------

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # ------------------------------------------------
            # Keep box inside frame
            # ------------------------------------------------

            x1 = max(
                0,
                min(x1, width - 1)
            )

            y1 = max(
                0,
                min(y1, height - 1)
            )

            x2 = max(
                0,
                min(x2, width - 1)
            )

            y2 = max(
                0,
                min(y2, height - 1)
            )


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # DRAW ID
            # =================================================

            if track_ids[i] is not None:

                person_id = int(track_ids[i])

                label = (
                    f"ID: {person_id} "
                    f"Conf: {confidence:.2f}"
                )

            else:

                label = (
                    f"Conf: {confidence:.2f}"
                )


            # =================================================
            # LABEL BACKGROUND
            # =================================================

            (text_width, text_height), baseline = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )
            )


            label_y = max(
                y1,
                text_height + baseline + 5
            )


            cv2.rectangle(
                annotated_frame,
                (
                    x1,
                    label_y - text_height - baseline - 5
                ),
                (
                    x1 + text_width + 5,
                    label_y
                ),
                (0, 255, 0),
                -1
            )


            # =================================================
            # DRAW LABEL
            # =================================================

            cv2.putText(
                annotated_frame,
                label,
                (
                    x1 + 2,
                    label_y - 4
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )


    # ========================================================
    # COUNT CURRENT TRACKS
    # ========================================================

    current_count = 0

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        current_count = len(result.boxes.id)


    # ========================================================
    # DISPLAY CURRENT COUNT
    # ========================================================

    count_text = f"People: {current_count}"

    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (230, 60),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated_frame,
        count_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY FRAME NUMBER
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE TRACKED FRAME
    # ========================================================

    output_frame_path = os.path.join(
        TRACKED_FRAMES_DIR,
        f"frame_{frame_number:06d}.jpg"
    )

    cv2.imwrite(
        output_frame_path,
        annotated_frame
    )


    processed_frame_count += 1


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "YOLO + BoT-SORT",
        annotated_frame
    )


    # ========================================================
    # PRESS Q TO STOP
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print()
        print("Processing stopped by user.")

        break


# ============================================================
# CLOSE DISPLAY
# ============================================================

cv2.destroyAllWindows()


# ============================================================
# 8. CREATE OUTPUT VIDEO
# ============================================================

print()
print("======================================")
print("STEP 3: CREATING OUTPUT VIDEO")
print("======================================")


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


processed_frame_number = 0


while True:

    frame_path = os.path.join(
        TRACKED_FRAMES_DIR,
        f"frame_{processed_frame_number:06d}.jpg"
    )


    if not os.path.exists(frame_path):
        break


    frame = cv2.imread(
        frame_path
    )


    if frame is None:
        break


    out.write(
        frame
    )


    processed_frame_number += 1


out.release()


# ============================================================
# 9. FINISHED
# ============================================================

print()
print("======================================")
print("PROCESSING COMPLETE")
print("======================================")

print(
    "Original frames :",
    FRAMES_DIR
)

print(
    "Processed frames:",
    TRACKED_FRAMES_DIR
)

print(
    "Output video    :",
    OUTPUT_VIDEO
)

print(
    "Frames written  :",
    processed_frame_number
)

print()
print("Done.")
