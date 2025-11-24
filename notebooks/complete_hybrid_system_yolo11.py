import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from collections import defaultdict

# Import your components

from hybrid_grid_calibrator import HybridGridCalibrator

def main():
    """
    Main function demonstrating complete workflow with YOLOv11
    """
    print("="*70)
    print("HYBRID CALIBRATION SYSTEM WITH YOLOv11n-seg")
    print("="*70)
    
    # Configuration
    calibration_video = "video/Traffic_3.mp4"
    processing_video = "video/Traffic_3.mp4"
    calibration_file = "hybrid_calibration_yolo11.json"
    
    # Step 1: Setup calibration
    print("\n📋 Step 1: Camera Calibration with YOLOv11")
    print("-" * 70)
    
    calibrator = HybridGridCalibrator(
        num_lanes=4,
        num_depth_zones=3,
        image_size=(640, 640)
    )
    
    if os.path.exists(calibration_file):
        print("Found existing calibration file")
        calibrator.load_calibration(calibration_file)
    else:
        print("Performing first-time calibration...")
        calibrator.auto_calibrate(calibration_video)
        calibrator.save_calibration(calibration_file)
    
    # Step 2: Process video with YOLOv11
    print("\n📹 Step 2: Processing Traffic Video with YOLOv11n-seg")
    print("-" * 70)
    
    # Load YOLOv11n-seg model
    detector = YOLO('yolo11n-seg.pt')  # UPDATED!
    cap = cv2.VideoCapture(processing_video)
    
    frame_count = 0
    vehicle_sizes = []
    processing_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time the inference
        start_time = cv2.getTickCount()
        
        # Detect vehicles with YOLOv11
        results = detector(frame, classes=[2, 5, 7], verbose=False)
        
        # Calculate inference time
        inference_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
        processing_times.append(inference_time)
        
        for result in results:
            if result.masks is not None:
                for mask, box in zip(result.masks.xy, result.boxes):
                    # Get vehicle position
                    x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                    y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                    
                    # Estimate size using hybrid calibration
                    mask_np = mask.astype(np.int32)
                    size = calibrator.estimate_size(mask_np, x_center, y_center)
                    
                    vehicle_sizes.append(size)
                    
                    # Visualize
                    cv2.polylines(frame, [mask_np], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"{size:.1f}m^2", 
                               (int(x_center), int(y_center)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display FPS and model info
        cv2.putText(frame, f"YOLOv11n-seg | {inference_time:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display
        cv2.imshow('YOLOv11 Hybrid Calibration System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            avg_time = np.mean(processing_times[-30:]) if processing_times else 0
            print(f"   Processed {frame_count} frames | "
                  f"Vehicles: {len(vehicle_sizes)} | "
                  f"Avg inference: {avg_time:.1f}ms")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Step 3: Results summary
    print("\n📊 Step 3: Results Summary")
    print("-" * 70)
    print(f"Total frames processed: {frame_count}")
    print(f"Total vehicles measured: {len(vehicle_sizes)}")
    
    if processing_times:
        print(f"\nYOLOv11n-seg Performance:")
        print(f"  Avg inference time: {np.mean(processing_times):.2f} ms")
        print(f"  FPS capability: {1000/np.mean(processing_times):.1f}")
    
    if vehicle_sizes:
        print(f"\nVehicle Size Statistics:")
        print(f"  Mean:   {np.mean(vehicle_sizes):.2f} m²")
        print(f"  Median: {np.median(vehicle_sizes):.2f} m²")
        print(f"  Std:    {np.std(vehicle_sizes):.2f} m²")
        print(f"  Min:    {np.min(vehicle_sizes):.2f} m²")
        print(f"  Max:    {np.max(vehicle_sizes):.2f} m²")
    
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE WITH YOLOv11!")
    print("="*70)

if __name__ == "__main__":
    main()
