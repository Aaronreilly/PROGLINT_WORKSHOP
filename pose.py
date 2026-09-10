import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = "bag4.mp4"
BAG_CONF_THRESH = 0.05
KP_CONF_THRESH = 0.1          # lowered further — wrist conf drops when gripping something

BAG_CLASS_IDS = [24, 26, 28]   # backpack, handbag, suitcase — ONLY these
PERSON_CLASS_ID = [0]          # person — ONLY this

# ---------------- MODELS ----------------
# using the larger "l" models — slower per frame but noticeably more accurate
# detection/pose than the "n" (nano) models, worth it if your machine can keep up
pose_model = YOLO("yolov8l-pose.pt")
bag_model = YOLO("yolov8l.pt")

# ---------------- SKELETON ----------------
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SKELETON_EDGES = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
]

# ---------------- STATE ----------------
# 5-state cycle, driven purely by wrist-vs-ROI and bag-vs-ROI (no proximity/touch logic needed)
#   BAG_THERE -> "Bag is there"     : idle, bag resting in ROI
#   PICKING   -> "picking the bag"  : wrist has entered the ROI
#   PICKED    -> "bag picked"       : wrist (carrying bag) has left the ROI
#   PLACING   -> "placing"          : wrist + bag are back inside the ROI
#   PLACED    -> "placed"           : bag is back in ROI, wrist has left
state = "BAG_THERE"
last_bag_box = None

LABELS = {
    "BAG_THERE": "Bag is there",
    "PICKING": "picking the bag",
    "PICKED": "bag picked",
    "PLACING": "placing",
    "PLACED": "placed",
}

# ---------------- HELPERS ----------------
def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def box_diagonal(box):
    return np.hypot(box[2] - box[0], box[3] - box[1])

def in_roi(box, roi):
    cx, cy = box_center(box)
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]

def any_wrist_in_roi(pose_res, roi):
    """True if ANY detected wrist (confident enough) currently sits inside the ROI."""
    if pose_res.keypoints is None:
        return False
    for kp in pose_res.keypoints.data:
        kp = kp.tolist()
        for idx in [L_WRIST, R_WRIST]:
            x, y, c = kp[idx]
            if c < KP_CONF_THRESH:
                continue
            if roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]:
                return True
    return False

def draw_skeleton(frame, keypoints):
    for (x, y, c) in keypoints:
        if c > KP_CONF_THRESH:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    for a, b in SKELETON_EDGES:
        if keypoints[a][2] > KP_CONF_THRESH and keypoints[b][2] > KP_CONF_THRESH:
            pa = (int(keypoints[a][0]), int(keypoints[a][1]))
            pb = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pa, pb, (0, 200, 255), 2)

def draw_label_on_box(frame, box, text, color=(0, 255, 255)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# ---------------- VIDEO SETUP ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

# ---- AUTOMATIC ROI: scan the first few frames, pick the most confident bag detection ----
ROI_PADDING = 40
AUTO_ROI_CONF_THRESH = 0.35   # MUCH stricter than tracking conf — a bad initial ROI ruins everything downstream
AUTO_ROI_SCAN_FRAMES = 15     # check multiple early frames instead of trusting frame 1 alone

best_box = None
best_conf = 0
scan_frame = first_frame
for i in range(AUTO_ROI_SCAN_FRAMES):
    res = bag_model(scan_frame, verbose=False, conf=AUTO_ROI_CONF_THRESH, classes=BAG_CLASS_IDS)[0]
    for box in res.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_box = box.xyxy[0].tolist()
            best_conf = conf
    ret, scan_frame = cap.read()
    if not ret:
        break

if best_box is not None:
    fh, fw = first_frame.shape[:2]
    x1 = max(0, int(best_box[0]) - ROI_PADDING)
    y1 = max(0, int(best_box[1]) - ROI_PADDING)
    x2 = min(fw, int(best_box[2]) + ROI_PADDING)
    y2 = min(fh, int(best_box[3]) + ROI_PADDING)
    roi = (x1, y1, x2, y2)
    print(f"[auto-ROI] bag detected (conf={best_conf:.2f}) -> ROI = {roi}")

    # ---- SHOW the detected ROI on screen and WAIT for a keypress before continuing ----
    # this is your visual proof of exactly where auto-ROI locked on, and at what confidence
    preview = first_frame.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(preview, f"AUTO ROI (conf={best_conf:.2f}) - press any key to start",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Auto-ROI Preview - press any key to continue", preview)
    cv2.waitKey(0)
    cv2.destroyWindow("Auto-ROI Preview - press any key to continue")
else:
    # fallback: no bag found in the scanned frames, ask user to draw it manually
    print("[auto-ROI] no bag detected in first frames — falling back to manual selection")
    roi_box = cv2.selectROI("Select ROI - press ENTER when done", first_frame, False, False)
    cv2.destroyWindow("Select ROI - press ENTER when done")
    x, y, w, h = roi_box
    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected — you must drag a box before pressing ENTER")
    roi = (x, y, x + w, y + h)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------------- MAIN LOOP ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # pose model only ever detects "person" by nature of the model, so no class filter needed
    pose_res = pose_model(frame, verbose=False)[0]
    # ONLY bag classes
    bag_res = bag_model(frame, verbose=False, conf=BAG_CONF_THRESH, classes=BAG_CLASS_IDS)[0]

    # ---- draw skeleton + person bounding box ----
    if pose_res.keypoints is not None:
        for kp in pose_res.keypoints.data:
            draw_skeleton(frame, kp.tolist())

    if pose_res.boxes is not None and len(pose_res.boxes) > 0:
        for pbox in pose_res.boxes.xyxy.tolist():
            draw_label_on_box(frame, pbox, "person", color=(0, 255, 0))
    else:
        cv2.putText(frame, "person: NOT DETECTED", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ---- best bag ----
    bag_box = None
    best_conf = 0
    for box in bag_res.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            bag_box = box.xyxy[0].tolist()
            best_conf = conf

    if bag_box is not None:
        last_bag_box = bag_box

    bag_in_roi = bag_box is not None and in_roi(bag_box, roi)
    wrist_in_roi = any_wrist_in_roi(pose_res, roi)

    # ---- state machine (pure ROI-crossing logic) ----
    if state in ("BAG_THERE", "PLACED") and wrist_in_roi:
        state = "PICKING"

    elif state == "PICKING":
        if not wrist_in_roi:
            state = "PICKED"

    elif state == "PICKED" and wrist_in_roi and bag_in_roi:
        state = "PLACING"

    elif state == "PLACING":
        if not wrist_in_roi:
            state = "PLACED"

    # ---- draw ROI + bag label ----
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
    box_to_label = bag_box if bag_box is not None else last_bag_box
    if box_to_label is not None:
        draw_label_on_box(frame, box_to_label, LABELS[state])

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
