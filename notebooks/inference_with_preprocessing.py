"""
Example integration of video preprocessor with RF-DETR inference
"""

import cv2
from pathlib import Path
from PIL import Image
import supervision as sv
from video_preprocessor import ComprehensiveVideoPreprocessor, ScenarioType
from roboflow import Roboflow
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights="output/checkpoint_best_total.pth")
model.optimize_for_inference()

rf = Roboflow(api_key="sjKAVuO8Lkaq5h2dDfDA")
project = rf.workspace("fyp-vfrgn").project("veiculos-contar-dnosk")
version = project.version(2)
dataset = version.download("coco")

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/test",
    annotations_path=f"{dataset.location}/test/_annotations.coco.json",
)

# Your existing model setup
NUM_CLASSES = len(ds.classes)
class_names = ds.classes

SOURCE_VIDEO_PATH = "video/Traffic_3.mp4"
TARGET_VIDEO_PATH = Path("output/output_video/output_Traffic_3_preprocessed_712x412.mp4")
CONFIDENCE_THRESHOLD = 0.5

# Ensure the output directory exists
TARGET_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

# Color palette for annotations
color_palette = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
    "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

# Initialize model
model = RFDETRBase(
    num_classes=NUM_CLASSES,
    pretrain_weights="output/checkpoint_best_total.pth"
)
model.optimize_for_inference()

# Initialize annotators
box_annotator = sv.BoxAnnotator(color=color_palette, thickness=2)
label_annotator = sv.LabelAnnotator(
    color=color_palette,
    text_color=sv.Color.BLACK,
    text_scale=0.5,
    text_thickness=1,
)

# ===========================================
# INITIALIZE VIDEO PREPROCESSOR
# ===========================================

# Option 1: Auto-detect scenario for each frame (recommended for varying conditions)
preprocessor = ComprehensiveVideoPreprocessor(auto_detect_scenario=True)

# Option 2: Manual scenario selection (if you know your video conditions)
# preprocessor = ComprehensiveVideoPreprocessor(
#     auto_detect_scenario=False, 
#     manual_scenario=ScenarioType.NIGHTTIME
# )

# Option 3: Use the convenience function
# from video_preprocessor import create_preprocessor
# preprocessor = create_preprocessor(scenario='nighttime', auto_detect=False)

# ===========================================

# Video capture setup
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open video: {SOURCE_VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing video: {SOURCE_VIDEO_PATH}")
print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(TARGET_VIDEO_PATH), fourcc, fps, (width, height))

# Statistics tracking
scenario_counts = {scenario: 0 for scenario in ScenarioType}
frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ===========================================
        # APPLY PREPROCESSING
        # ===========================================
        processed_frame, detected_scenario = preprocessor.process_frame(frame)
        scenario_counts[detected_scenario] += 1
        
        # Log scenario changes
        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}/{total_frames} - Scenario: {detected_scenario.value}")
        
        # ===========================================
        # RUN DETECTION ON PREPROCESSED FRAME
        # ===========================================
        target_size = (712, 412)  # (width, height)
        resized_frame = cv2.resize(processed_frame, target_size, interpolation=cv2.INTER_LINEAR)

        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)
        
        # Create labels with confidence scores
        labels = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        # Annotate the preprocessed frame
        annotated_frame = box_annotator.annotate(scene=processed_frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )
        
        # Optional: Add scenario information to frame
        cv2.putText(
            annotated_frame,
            f"Scenario: {detected_scenario.value}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Write the annotated frame
        writer.write(annotated_frame)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
            
finally:
    cap.release()
    writer.release()
    
    print("\n" + "="*50)
    print(f"Inference complete. Saved annotated video to {TARGET_VIDEO_PATH}")
    print(f"\nScenario distribution across {frame_idx} frames:")
    for scenario, count in scenario_counts.items():
        if count > 0:
            percentage = (count / frame_idx) * 100
            print(f"  {scenario.value}: {count} frames ({percentage:.1f}%)")


# ===========================================
# ALTERNATIVE: Process specific scenarios only
# ===========================================
"""
# If you want to apply different preprocessing based on video segments:

def process_video_with_time_based_scenarios(video_path, scenario_schedule):
    '''
    Process video with different scenarios at different time segments.
    
    Args:
        video_path: Path to input video
        scenario_schedule: List of (start_time, end_time, scenario) tuples
        
    Example:
        scenario_schedule = [
            (0, 30, ScenarioType.DAYTIME),      # First 30 seconds
            (30, 60, ScenarioType.NIGHTTIME),   # 30-60 seconds
            (60, None, ScenarioType.RAINY_FOGGY) # Rest of video
        ]
    '''
    preprocessor = ComprehensiveVideoPreprocessor(auto_detect_scenario=False)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = frame_idx / fps
        
        # Determine scenario based on schedule
        for start, end, scenario in scenario_schedule:
            if start <= current_time < (end or float('inf')):
                preprocessor.manual_scenario = scenario
                break
        
        processed_frame, _ = preprocessor.process_frame(frame)
        # ... rest of processing ...
        
        frame_idx += 1
"""

# ===========================================
# TIPS FOR OPTIMAL PERFORMANCE
# ===========================================
"""
Performance optimization tips:

1. **Batch Processing**: Process frames in batches if your model supports it
2. **Skip Frames**: For real-time applications, process every Nth frame
3. **Region of Interest**: Apply preprocessing only to ROI if applicable
4. **GPU Acceleration**: Use cv2.cuda functions if available
5. **Parallel Processing**: Use multiprocessing for preprocessing pipeline

Example of frame skipping:
    if frame_idx % 2 == 0:  # Process every other frame
        processed_frame, scenario = preprocessor.process_frame(frame)
    else:
        processed_frame = frame  # Use original for skipped frames
"""
