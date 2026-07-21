import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict

# ----------------- Centroid Tracker -----------------
class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in counted_ids:
            del counted_ids[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in set(range(D.shape[1])) - used_cols:
                self.register(input_centroids[col])

        return self.objects

# ---------------- Model and Video ----------------
model = YOLO("yolov8s.pt")  # ⬅️ Use yolov8s or yolov8m for much better accuracy

cap = cv2.VideoCapture("test.mp4")
ct = CentroidTracker()
track_history = {}
counted_ids = {}

up_count, down_count = 0, 0

# Read first frame
ret, frame = cap.read()
if not ret:
    raise Exception("Cannot read video.")
H, W = frame.shape[:2]
line_y = H // 2

# --------------- IOU Function ----------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --------------- Processing Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    centroids = []
    boxes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.6:  # More strict confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Apply IOU filtering to reduce duplicate boxes
                skip = False
                for b in boxes:
                    if compute_iou([x1, y1, x2, y2], b) > 0.5:
                        skip = True
                        break
                if skip:
                    continue

                centroids.append((cx, cy))
                boxes.append([x1, y1, x2, y2])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    objects = ct.update(np.array(centroids))

    for object_id, (cx, cy) in objects.items():
        if object_id not in track_history:
            track_history[object_id] = []
        track_history[object_id].append((cx, cy))

        # Smooth short history (reduce miscount from shaky tracks)
        if len(track_history[object_id]) > 10:
            track_history[object_id] = track_history[object_id][-10:]

        if len(track_history[object_id]) >= 2:
            prev_cy = track_history[object_id][-2][1]
            direction = cy - prev_cy

            if prev_cy < line_y and cy >= line_y and object_id not in counted_ids:
                down_count += 1
                counted_ids[object_id] = 'down'
                print(f"[DOWN] ID {object_id}")
            elif prev_cy > line_y and cy <= line_y and object_id not in counted_ids:
                up_count += 1
                counted_ids[object_id] = 'up'
                print(f"[UP] ID {object_id}")

        cv2.putText(frame, f"ID {object_id}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw center line
    cv2.line(frame, (0, line_y), (W, line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"IN: {up_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"OUT: {down_count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
