import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = "bag4.mp4"
BAG_CONF_THRESH = 0.05
KP_CONF_THRESH = 0.1          # lowered further — wrist conf drops when gripping something

WRIST_DIST_MULTIPLIER = 1.2   # threshold = bag's own diagonal size * this — scales with distance-to-camera

BAG_CLASS_IDS = [24, 26, 28]   # backpack, handbag, suitcase — ONLY these
PERSON_CLASS_ID = [0]          # person — ONLY this

# ---------------- MODELS ----------------
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
state = "BAG IS KEPT"
last_bag_box = None

# ---------------- HELPERS ----------------
def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def box_diagonal(box):
    return np.hypot(box[2] - box[0], box[3] - box[1])

def in_roi(box, roi):
    cx, cy = box_center(box)
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]

def boxes_overlap(a, b):
    """True if two xyxy boxes intersect at all."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def wrist_near_bag(keypoints, bag_box, thresh):
    bx, by = box_center(bag_box)
    for idx in [L_WRIST, R_WRIST]:
        x, y, c = keypoints[idx]
        if c < KP_CONF_THRESH:
            continue
        if np.hypot(x - bx, y - by) < thresh:
            return True
    return False

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

# ---- AUTOMATIC ROI: detect the bag in the first frame, use its box (+padding) as ROI ----
ROI_PADDING = 40   # pixels of breathing room around the detected bag

auto_res = bag_model(first_frame, verbose=False, conf=BAG_CONF_THRESH, classes=BAG_CLASS_IDS)[0]

best_box = None
best_conf = 0
for box in auto_res.boxes:
    conf = float(box.conf[0])
    if conf > best_conf:
        best_box = box.xyxy[0].tolist()
        best_conf = conf

if best_box is not None:
    fh, fw = first_frame.shape[:2]
    x1 = max(0, int(best_box[0]) - ROI_PADDING)
    y1 = max(0, int(best_box[1]) - ROI_PADDING)
    x2 = min(fw, int(best_box[2]) + ROI_PADDING)
    y2 = min(fh, int(best_box[3]) + ROI_PADDING)
    roi = (x1, y1, x2, y2)
    print(f"[auto-ROI] bag detected in frame 1 (conf={best_conf:.2f}) -> ROI = {roi}")
else:
    # fallback: no bag found in frame 1, ask user to draw it manually
    print("[auto-ROI] no bag detected in first frame — falling back to manual selection")
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

    # ONLY person
    pose_res = pose_model(frame, verbose=False, classes=PERSON_CLASS_ID)[0]
    # ONLY bag classes
    bag_res = bag_model(frame, verbose=False, conf=BAG_CONF_THRESH, classes=BAG_CLASS_IDS)[0]

    # ---- draw skeleton ----
    if pose_res.keypoints is not None:
        for kp in pose_res.keypoints.data:
            draw_skeleton(frame, kp.tolist())

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

    # ---- proximity check: wrist distance (scaled to bag size) OR bbox overlap ----
    person_near_bag = False
    if bag_box is not None:
        wrist_thresh = box_diagonal(bag_box) * WRIST_DIST_MULTIPLIER

        # signal 1: wrist close to bag center
        if pose_res.keypoints is not None:
            for kp in pose_res.keypoints.data:
                if wrist_near_bag(kp.tolist(), bag_box, wrist_thresh):
                    person_near_bag = True
                    break

        # signal 2: person's bounding box physically overlaps the bag's bounding box
        if not person_near_bag and pose_res.boxes is not None:
            for pbox in pose_res.boxes.xyxy.tolist():
                if boxes_overlap(pbox, bag_box):
                    person_near_bag = True
                    break

    bag_in_roi = bag_box is not None and in_roi(bag_box, roi)
    wrist_in_roi = any_wrist_in_roi(pose_res, roi)

    # ---- state machine ----
    if state == "BAG IS KEPT" and person_near_bag:
        state = "PICKING BAG"

    elif state == "PICKING BAG":
        # switched to wrist-vs-ROI instead of bag-vs-ROI —
        # the bag itself often gets occluded by the arm/body while being carried,
        # but the wrist keypoint keeps tracking reliably
        if not wrist_in_roi:
            state = "BAG IS PICKED"
        elif not person_near_bag:
            state = "BAG IS KEPT"

    elif state == "BAG IS PICKED" and wrist_in_roi:
        state = "PLACING BAG"

    elif state == "PLACING BAG":
        # bag confirmed back in ROI -> done
        if bag_box is not None and bag_in_roi:
            state = "BAG IS KEPT"
        # OR: hand is back inside the ROI and has let go of the bag
        # (covers the case where the bag is briefly occluded right after being set down)
        elif wrist_in_roi and not person_near_bag:
            state = "BAG IS KEPT"
        elif not wrist_in_roi:
            state = "BAG IS PICKED"

    # ---- safety net ----
    # if the bag is plainly sitting inside the ROI right now and nobody is
    # touching it, the ground truth is "kept" no matter what the state machine
    # above concluded (covers cases where the wrist never re-enters the ROI,
    # e.g. person lets go and walks off camera instead of stepping back through it)
    if bag_box is not None and bag_in_roi and not person_near_bag:
        state = "BAG IS KEPT"

    # ---- draw ROI + bag label ----
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
    box_to_label = bag_box if bag_box is not None else last_bag_box
    if box_to_label is not None:
        draw_label_on_box(frame, box_to_label, state)

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
