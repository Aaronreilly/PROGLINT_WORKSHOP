import cv2
import os
from ultralytics import YOLO

def run_pure_tracker(video_filename="test.mp4"):
    if not os.path.exists(video_filename):
        print(f"Error: The file '{video_filename}' was not found.")
        return

    model = YOLO("yolo11n.pt") 
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print(f"Error: OpenCV could not open '{video_filename}'.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    output_filename = "tracked_output.mp4"
    video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print("Tracking engine running...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()     
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) 

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        video_writer.write(frame)
        cv2.imshow("Proglint Solutions - Pure ID Tracking Window", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Saved tracking output to: '{output_filename}'")

if __name__ == "__main__":
    run_pure_tracker(video_filename="test.mp4")
