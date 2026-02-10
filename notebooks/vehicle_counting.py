import cv2
import os
from pathlib import Path
from PIL import Image
import supervision as sv
import math
from roboflow import Roboflow
from rfdetr import RFDETRBase
import supervision as sv
# rf = Roboflow(api_key="sjKAVuO8Lkaq5h2dDfDA")
# project = rf.workspace("fyp-vfrgn").project("veiculos-contar-dnosk")
# version = project.version(3)
# dataset = version.download("coco")

model = RFDETRBase(pretrain_weights="output/checkpoint_best_total.pth")
model.optimize_for_inference()

# ds = sv.DetectionDataset.from_coco(
#     images_directory_path=f"{dataset.location}/test",
#     annotations_path=f"{dataset.location}/test/_annotations.coco.json",
# )
# --- Setup ---
SOURCE_VIDEO_PATH = "video/daytime.mp4"
CONFIDENCE_THRESHOLD = 0.5
width, height = 672, 448

# Use your dataset class names
# class_names = ds.classes
class_names = ['cars-counter', 'Bus', 'Motorcycle', 'Pickup', 'Sedan', 'Suv', 'Truck', 'Van']

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output_dir = "output/output_video"
os.makedirs(output_dir, exist_ok=True)
original_filename = os.path.splitext(os.path.basename(SOURCE_VIDEO_PATH))[0]
output_filename = f"{original_filename}_counted.mp4"
output_path = os.path.join(output_dir, output_filename)
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# --- Line drawing ---
line_points = []

def draw_line(event, x, y, flags, param):
    global line_points
    base = param.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        line_points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and len(line_points) == 1:
        cv2.line(base, line_points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Define Line", base)
        return
    elif event == cv2.EVENT_LBUTTONUP and len(line_points) == 1:
        line_points.append((x, y))
        cv2.line(base, line_points[0], line_points[1], (0, 255, 0), 2)
        cv2.imshow("Define Line", base)
        return
    cv2.imshow("Define Line", base)

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Unable to read video")
first_frame = cv2.resize(first_frame, (width, height))
cv2.imshow("Define Line", first_frame)
cv2.setMouseCallback("Define Line", draw_line, first_frame)
print("Draw a line by click-and-dragging on the frame window...")
cv2.waitKey(0)
cv2.destroyWindow("Define Line")

if len(line_points) != 2:
    raise RuntimeError("Line not defined. Please click and drag to draw a line.")

# --- ByteTrack tracker ---
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,  # min confidence to start a track
    lost_track_buffer=30,             # frames to keep lost tracks
    minimum_matching_threshold=0.8,   # IoU threshold for matching
    frame_rate=int(fps)
)

# --- Counting setup ---
vehicle_counts = {}          # {class_name: count}
track_class = {}             # {track_id: class_id} - remember class for each track
track_last_dist = {}         # {track_id: last signed distance to line}
counted_track_ids = set()    # track IDs already counted

DIST_THRESHOLD = 25.0        # pixels band around line for crossing

# --- Annotators ---
resolution_wh = (width, height)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080", "#b266ff"
])

bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
label_annotator = sv.LabelAnnotator(
    color=color,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    smart_position=True,
    text_padding=5
)
trace_annotator = sv.TraceAnnotator(color=color, thickness=thickness)

# --- Helper: signed distance to line (in pixels) + projection check ---
def line_signed_distance(p1, p2, centroid):
    """Returns (signed_distance, is_within_segment)"""
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = centroid

    # Line vector
    dx = x2 - x1
    dy = y2 - y1
    line_len_sq = dx * dx + dy * dy

    if line_len_sq == 0:
        return 0.0, False

    # Parameter t: projection of centroid onto line (0 = at p1, 1 = at p2)
    t = ((cx - x1) * dx + (cy - y1) * dy) / line_len_sq

    # Check if projection falls within segment (with small margin)
    margin = 0.1  # allow 10% beyond endpoints
    is_within_segment = -margin <= t <= 1.0 + margin

    # Signed distance from point to infinite line
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    denom = math.hypot(a, b)
    signed_dist = (a * cx + b * cy + c) / denom if denom != 0 else 0.0

    return signed_dist, is_within_segment

# --- Main loop ---
frame_idx = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)

        # --- Update tracker ---
        detections = tracker.update_with_detections(detections)

        # Build labels with track ID
        labels = [
            f"#{tracker_id} {class_names[class_id]} {confidence:.2f}"
            for tracker_id, class_id, confidence 
            in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]

        # Annotate
        annotated_frame = resized_frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Draw counting line
        cv2.line(annotated_frame, line_points[0], line_points[1], (0, 255, 0), 2)

        # --- Count vehicles crossing ---
        for tracker_id, class_id, xyxy in zip(
            detections.tracker_id, detections.class_id, detections.xyxy
        ):
            if tracker_id is None:
                continue

            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)

            dist, is_within = line_signed_distance(line_points[0], line_points[1], (cx, cy))
            prev_data = track_last_dist.get(tracker_id)

            # Remember class for this track
            track_class[tracker_id] = int(class_id)

            # Check crossing: side changed, close to line, AND within segment bounds
            if prev_data is not None and tracker_id not in counted_track_ids:
                prev_dist, prev_within = prev_data
                # Only count if currently or previously within the segment bounds
                if (prev_dist * dist < 0 and 
                    min(abs(prev_dist), abs(dist)) < DIST_THRESHOLD and
                    (is_within or prev_within)):
                    cls_name = class_names[track_class[tracker_id]]
                    vehicle_counts[cls_name] = vehicle_counts.get(cls_name, 0) + 1
                    counted_track_ids.add(tracker_id)
                    print(f"[Track {tracker_id}] {cls_name} crossed line. Total: {vehicle_counts[cls_name]}")

            track_last_dist[tracker_id] = (dist, is_within)

        # Overlay counts
        y_offset = 30
        for cls, count in vehicle_counts.items():
            cv2.putText(annotated_frame, f"{cls}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

        writer.write(annotated_frame)
        cv2.imshow("RF-DETR Counting (ByteTrack)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("Final vehicle counts:", vehicle_counts)
print(f"Inference complete. Saved counted video to {output_path}")