import numpy as np
import cv2
from ultralytics import YOLO
import json
from collections import defaultdict

from simplified_vp_detector import SimplifiedVPDetector

class HybridGridCalibrator:
    """
    Hybrid calibration: Automatic VP detection + grid population
    NOW UPDATED FOR YOLOv11n-seg
    """
    
    def __init__(self, num_lanes=3, num_depth_zones=3, image_size=(640, 640)):
        """
        Initialize the calibrator with YOLOv11
        
        Args:
            num_lanes: Number of traffic lanes
            num_depth_zones: Number of depth regions (near/mid/far)
            image_size: Input image size for YOLOv11 processing
        """
        self.num_lanes = num_lanes
        self.num_depth_zones = num_depth_zones
        self.image_width, self.image_height = image_size
        
        # Initialize simplified VP detector
        self.vp_detector = SimplifiedVPDetector()
        
        # Grid to store pixels-per-meter values
        # Key: (lane_idx, depth_idx), Value: pixels-per-meter
        self.grid = {}
        
        # YOLOv11n-seg model (UPDATED!)
        print("Loading YOLOv11n-seg model...")
        self.detector = YOLO('yolo11n-seg.pt')  # Changed from yolov8n-seg.pt
        print("✅ YOLOv11n-seg loaded successfully")
        
        # Statistics for refinement
        self.vehicle_stats = []
        
        # Expected vehicle dimensions (meters)
        self.VEHICLE_DIMS = {
            'car': {'length': (3.5, 5.5), 'width': (1.6, 2.2), 'area': (5.6, 12.1)},
            'truck': {'length': (5.0, 12.0), 'width': (2.0, 2.6), 'area': (10.0, 31.2)},
            'bus': {'length': (10.0, 15.0), 'width': (2.3, 2.6), 'area': (23.0, 39.0)}
        }
    
    def auto_calibrate(self, video_path, num_frames=200):
        """
        Perform automatic calibration using YOLOv11
        
        Args:
            video_path: Path to calibration video
            num_frames: Number of frames to process
        """
        print("\n" + "="*70)
        print("🔧 AUTOMATIC CALIBRATION WITH YOLOv11n-seg")
        print("="*70)
        
        # Step 1: Detect vanishing point
        print("\n📍 Step 1: Detecting Vanishing Point")
        print("-" * 70)
        self.vp1 = self.vp_detector.detect_traffic_direction(video_path, num_frames=150)
        
        # Step 2: Initialize grid from perspective
        print("\n📐 Step 2: Computing Initial Grid from Perspective")
        print("-" * 70)
        self._compute_grid_from_perspective()
        
        # Step 3: Refine using vehicle detections with YOLOv11
        print("\n🚗 Step 3: Refining with YOLOv11 Vehicle Detections")
        print("-" * 70)
        self._refine_with_yolo11_detections(video_path, num_frames)
        
        print("\n" + "="*70)
        print("✅ CALIBRATION COMPLETE!")
        print("="*70)
        self._print_calibration_summary()
    
    def _refine_with_yolo11_detections(self, video_path, num_frames):
        """
        Refine calibration using YOLOv11n-seg detections
        """
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        vehicle_measurements = []
        
        print("🔍 Processing video with YOLOv11n-seg...")
        
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles with YOLOv11n-seg
            # Classes: 2=car, 5=bus, 7=truck
            results = self.detector(frame, classes=[2, 5, 7], verbose=False)
            
            for result in results:
                if result.masks is not None:
                    for mask, box, cls in zip(result.masks.xy, result.boxes.xyxy, result.boxes.cls):
                        # Get vehicle position
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        
                        # Get mask polygon
                        mask_np = mask.astype(np.int32)
                        
                        # Calculate pixel area
                        pixel_area = cv2.contourArea(mask_np)
                        
                        # Store measurement
                        vehicle_measurements.append({
                            'position': (x_center, y_center),
                            'pixel_area': pixel_area,
                            'class': int(cls),
                            'mask': mask_np
                        })
            
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"   Processed {frame_count}/{num_frames} frames, "
                      f"detected {len(vehicle_measurements)} vehicles")
        
        cap.release()
        
        print(f"✅ Detected {len(vehicle_measurements)} total vehicles with YOLOv11")
        
        # Statistical refinement
        if len(vehicle_measurements) > 10:
            self._statistical_refinement(vehicle_measurements)
        else:
            print("⚠️ Warning: Only {} vehicles detected. Keeping initial calibration.".format(
                len(vehicle_measurements)))
    
    def _statistical_refinement(self, measurements):
        """
        Refine grid using statistical analysis of YOLOv11 detections
        """
        print("\n📊 Performing statistical refinement...")
        
        # Group measurements by grid cell
        cell_measurements = defaultdict(list)
        
        for m in measurements:
            lane_idx = self._get_lane_index(m['position'][0], m['position'][1])
            depth_idx = self._get_depth_index(m['position'][1])
            
            if lane_idx is not None and depth_idx is not None:
                # Get expected area based on vehicle class
                if m['class'] == 2:  # car
                    expected_area = np.mean(self.VEHICLE_DIMS['car']['area'])
                elif m['class'] == 5:  # bus
                    expected_area = np.mean(self.VEHICLE_DIMS['bus']['area'])
                else:  # truck
                    expected_area = np.mean(self.VEHICLE_DIMS['truck']['area'])
                
                # Calculate PPM from area
                ppm_estimate = np.sqrt(m['pixel_area'] / expected_area)
                cell_measurements[(lane_idx, depth_idx)].append(ppm_estimate)
        
        # Update grid with refined values
        refinement_count = 0
        for (lane_idx, depth_idx), ppm_list in cell_measurements.items():
            if len(ppm_list) >= 3:  # Need at least 3 measurements
                # Use median for robustness
                refined_ppm = np.median(ppm_list)
                old_ppm = self.grid.get((lane_idx, depth_idx), refined_ppm)
                
                # Weighted average: 70% old, 30% new
                self.grid[(lane_idx, depth_idx)] = 0.7 * old_ppm + 0.3 * refined_ppm
                refinement_count += 1
        
        print(f"✅ Refined {refinement_count} grid cells using YOLOv11 detections")
    
    def _compute_grid_from_perspective(self):
        """Initialize grid based on vanishing point and perspective geometry"""
        # Similar implementation as before
        vp_x, vp_y = self.vp1
        
        # Baseline PPM values (adjusted for typical perspective)
        baseline_ppm = 40  # pixels per meter at image center
        
        # Initialize grid with perspective-based estimates
        for lane_idx in range(self.num_lanes):
            for depth_idx in range(self.num_depth_zones):
                # Lane position (normalized: 0=left, 1=right)
                lane_pos = lane_idx / max(1, self.num_lanes - 1)
                x_pos = lane_pos * self.image_width
                
                # Depth position (normalized: 0=near, 1=far)
                depth_pos = depth_idx / max(1, self.num_depth_zones - 1)
                y_pos = (1 - depth_pos) * self.image_height  # Inverted: near=bottom
                
                # Distance from vanishing point affects scale
                dist_to_vp = np.sqrt((x_pos - vp_x)**2 + (y_pos - vp_y)**2)
                max_dist = np.sqrt(self.image_width**2 + self.image_height**2)
                
                # Perspective scaling (objects smaller when closer to VP)
                perspective_factor = 1 + 2 * (1 - dist_to_vp / max_dist)
                
                # Set initial PPM
                self.grid[(lane_idx, depth_idx)] = baseline_ppm * perspective_factor
        
        print("✅ Grid initialized from perspective geometry")

    def _get_lane_index(self, x, y=None):
        """
        Get lane index using Vanishing Point projection.
        Maps point (x,y) to the bottom edge of the screen to find its lane.
        """
        if y is None: return 0 # Safety check
        
        # 1. Get Vanishing Point and Screen Dimensions
        vp_x, vp_y = self.vp1
        h, w = self.image_height, self.image_width
        
        # Avoid division by zero if point is at or above VP
        if y <= vp_y + 1: 
            return int(x / (w / self.num_lanes)) # Fallback to linear
            
        # 2. Project the point (x,y) onto the bottom line of the image (y=h)
        # Formula: derived from similar triangles
        # x_projected is where the ray from VP through (x,y) hits the bottom
        t = (h - vp_y) / (y - vp_y)
        x_projected = vp_x + (x - vp_x) * t
        
        # 3. Determine lane based on position at the bottom of the screen
        # We assume lanes are equally spaced at the camera's feet (bottom of image)
        lane_width_at_bottom = w / self.num_lanes
        lane_idx = int(x_projected / lane_width_at_bottom)
        
        return np.clip(lane_idx, 0, self.num_lanes - 1)

    def _get_depth_index(self, y):
        """
        Get depth index. 
        Note: For tilted views, strictly horizontal depth zones are usually 'good enough'
        unless the camera roll is extreme (>20 degrees).
        """
        # We stick to simple y-split for depth to keep stability, 
        # but you can adjust the split points if needed.
        depth_height = self.image_height / self.num_depth_zones
        depth_idx = int(y / depth_height)
        return np.clip(depth_idx, 0, self.num_depth_zones - 1)

    # def _get_lane_index(self, x):
    #     """Get lane index from x coordinate"""
    #     lane_width = self.image_width / self.num_lanes
    #     lane_idx = int(x / lane_width)
    #     return np.clip(lane_idx, 0, self.num_lanes - 1)
    
    # def _get_depth_index(self, y):
    #     """Get depth zone index from y coordinate"""
    #     depth_height = self.image_height / self.num_depth_zones
    #     depth_idx = int(y / depth_height)
    #     return np.clip(depth_idx, 0, self.num_depth_zones - 1)
    
    def estimate_size(self, mask_polygon, x_center, y_center):
        """
        Estimate real-world size of vehicle using YOLOv11 mask
        
        Args:
            mask_polygon: Vehicle mask polygon from YOLOv11
            x_center: Center x coordinate
            y_center: Center y coordinate
        
        Returns:
            Estimated area in square meters
        """
        lane_idx = self._get_lane_index(x_center, y_center)
        depth_idx = self._get_depth_index(y_center)
        
        # Get PPM for this grid cell
        ppm = self.grid.get((lane_idx, depth_idx), 40)  # Default: 40 ppm
        
        # Calculate pixel area
        pixel_area = cv2.contourArea(mask_polygon)
        
        # Convert to real-world area
        real_area = pixel_area / (ppm ** 2)
        
        return real_area
    
    def _print_calibration_summary(self):
        """Print calibration results"""
        print("\n📋 Calibration Summary:")
        print("-" * 50)
        print(f"Vanishing Point: ({self.vp1[0]:.1f}, {self.vp1[1]:.1f})")
        print(f"Grid Size: {self.num_lanes} lanes × {self.num_depth_zones} depth zones")
        print("\nPixels Per Meter (PPM) Grid:")
        
        for depth_idx in range(self.num_depth_zones):
            depth_name = ['Near', 'Mid', 'Far'][depth_idx] if depth_idx < 3 else f'Depth{depth_idx}'
            print(f"\n  {depth_name}:", end=' ')
            for lane_idx in range(self.num_lanes):
                ppm = self.grid.get((lane_idx, depth_idx), 0)
                print(f"Lane{lane_idx+1}={ppm:.1f}", end='  ')
    
    def save_calibration(self, filename):
        """Save calibration to file"""
        # Convert numpy types to Python types for JSON serialization
        calibration_data = {
            'vp1': [float(self.vp1[0]), float(self.vp1[1])],
            'grid': {f"{k[0]}_{k[1]}": float(v) for k, v in self.grid.items()},
            'num_lanes': int(self.num_lanes),
            'num_depth_zones': int(self.num_depth_zones),
            'model': 'yolo11n-seg'  # Updated model name
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n💾 Calibration saved to: {filename}")
    
    def load_calibration(self, filename):
        """Load calibration from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.vp1 = tuple(data['vp1'])
        self.num_lanes = int(data['num_lanes'])
        self.num_depth_zones = int(data['num_depth_zones'])
        
        # Convert string keys back to tuples
        self.grid = {}
        for key_str, value in data['grid'].items():
            lane_idx, depth_idx = map(int, key_str.split('_'))
            self.grid[(lane_idx, depth_idx)] = float(value)
        
        # Check model version
        if 'model' in data:
            if data['model'] != 'yolo11n-seg':
                print(f"⚠️ Warning: Calibration was done with {data['model']}, now using yolo11n-seg")
        
        print(f"✅ Calibration loaded from: {filename}")
        self._print_calibration_summary()
