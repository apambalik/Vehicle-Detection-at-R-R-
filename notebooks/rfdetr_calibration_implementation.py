"""
RF-DETR (Roboflow DETR) Based Automatic Camera Calibration for Traffic Monitoring
==================================================================================

This implementation uses RF-DETR (Roboflow's optimized DETR) for vehicle detection
while maintaining the automatic calibration approach from the Dubská et al. paper.

RF-DETR advantages:
- Optimized for production use
- Better small object detection
- Improved training efficiency
- Open source and actively maintained by Roboflow

Author: Implementation based on Dubská et al. paper with RF-DETR integration
"""

import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional, Dict
import json
import time
from dataclasses import dataclass
from collections import defaultdict

# For RF-DETR installation:
# pip install inference-gpu  # or inference for CPU
# pip install roboflow

try:
    from inference import get_model
except ImportError:
    print("Please install RF-DETR dependencies:")
    print("pip install inference-gpu  # For GPU support")
    print("pip install inference  # For CPU only")
    print("pip install roboflow supervision")


@dataclass
class VehicleDetection:
    """Data class for vehicle detections"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    class_id: int
    frame_id: int
    center_base: Tuple[int, int]
    corners: Dict[str, Tuple[int, int]]


class RFDETRVehicleDetector:
    """
    Vehicle detection using RF-DETR (Roboflow DETR) model
    """
    
    def __init__(self, confidence_threshold=0.35):
        """
        Initialize RF-DETR model
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        print("🚀 Initializing RF-DETR model...")
        
        # Initialize RF-DETR model (using COCO pretrained)
        # RF-DETR uses the inference SDK from Roboflow
        self.model = get_model("rfdetr-base")
        # Note: For local inference without API key, you can use:
        # self.model = get_model("coco/11", api_key="") 
        
        self.confidence_threshold = confidence_threshold
        
        # COCO vehicle classes that RF-DETR can detect
        self.vehicle_classes = {
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7
        }
        
        # Reverse mapping for class names
        self.class_id_to_name = {v: k for k, v in self.vehicle_classes.items()}
        
        # Track detections for temporal consistency
        self.detection_history = defaultdict(list)
        
    def detect_vehicles(self, frame: np.ndarray, frame_id: int = 0) -> List[VehicleDetection]:
        """
        Detect vehicles in frame using RF-DETR
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            frame_id: Frame number for tracking
            
        Returns:
            List of VehicleDetection objects
        """
        # Convert BGR to RGB for RF-DETR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run RF-DETR inference
        predictions = self.model.infer(rgb_frame)[0]
        
        vehicles = []
        
        # Process predictions
        if predictions and hasattr(predictions, 'predictions'):
            for pred in predictions.predictions:
                # Check if it's a vehicle class
                class_name = pred.class_name.lower()
                
                # RF-DETR returns class names, map to our vehicle types
                if class_name in self.vehicle_classes:
                    # Extract bounding box
                    x1 = int(pred.x - pred.width / 2)
                    y1 = int(pred.y - pred.height / 2)
                    x2 = int(pred.x + pred.width / 2)
                    y2 = int(pred.y + pred.height / 2)
                    
                    # Filter by confidence
                    if pred.confidence >= self.confidence_threshold:
                        # Estimate corners for 3D box construction
                        corners = self._estimate_vehicle_corners(x1, y1, x2, y2, class_name)
                        
                        detection = VehicleDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=pred.confidence,
                            class_name=class_name,
                            class_id=self.vehicle_classes[class_name],
                            frame_id=frame_id,
                            center_base=corners['center_base'],
                            corners=corners
                        )
                        
                        vehicles.append(detection)
        
        # Store in history for temporal analysis
        self.detection_history[frame_id] = vehicles
        
        return vehicles
    
    def _estimate_vehicle_corners(self, x1: int, y1: int, x2: int, y2: int, 
                                 vehicle_class: str) -> Dict[str, Tuple[int, int]]:
        """
        Estimate vehicle base corners from bounding box
        
        This is crucial for the 3D bounding box construction method from the paper.
        Since RF-DETR provides bounding boxes (not segmentation), we estimate
        the vehicle base corners using geometric assumptions.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            vehicle_class: Type of vehicle for class-specific adjustments
            
        Returns:
            Dictionary with corner points
        """
        # Base height ratio (how much of bbox is above ground)
        # Adjust based on vehicle type
        base_ratios = {
            'car': 0.85,      # Cars: wheels at ~85% of bbox height
            'bus': 0.90,      # Buses: higher ground clearance
            'truck': 0.88,    # Trucks: between car and bus
            'motorcycle': 0.80  # Motorcycles: lower profile
        }
        
        ratio = base_ratios.get(vehicle_class, 0.85)
        base_y = int(y1 + (y2 - y1) * ratio)
        
        # Account for perspective (vehicles farther appear more compressed)
        perspective_factor = 1.0 - (y1 / 1080.0) * 0.2  # Assuming 1080p video
        
        # Width adjustment for perspective
        width_adjustment = int((x2 - x1) * 0.05 * perspective_factor)
        
        corners = {
            'front_left': (x1 + width_adjustment, base_y),
            'front_right': (x2 - width_adjustment, base_y),
            'rear_left': (x1 + width_adjustment, int(base_y - (y2-y1)*0.1)),
            'rear_right': (x2 - width_adjustment, int(base_y - (y2-y1)*0.1)),
            'center_base': ((x1+x2)//2, base_y),
            'top_left': (x1, y1),
            'top_right': (x2, y1),
            'bbox': (x1, y1, x2, y2)
        }
        
        return corners
    
    def apply_nms(self, detections: List[VehicleDetection], iou_threshold: float = 0.5) -> List[VehicleDetection]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        
        RF-DETR sometimes produces overlapping detections, so NMS helps clean them up.
        
        Args:
            detections: List of vehicle detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if len(detections) <= 1:
            return detections
        
        # Convert to numpy array for easier manipulation
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Apply NMS
        indices = self._nms(boxes, scores, iou_threshold)
        
        return [detections[i] for i in indices]
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression implementation"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return keep


class VanishingPointDetector:
    """
    Vanishing point detection for camera calibration
    Simplified version that works well with RF-DETR detections
    """
    
    def __init__(self):
        self.motion_lines = []
        
    def detect_from_vehicle_tracks(self, detections_by_frame: Dict[int, List[VehicleDetection]], 
                                  frame_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Detect vanishing point from vehicle trajectories
        
        Args:
            detections_by_frame: Dictionary mapping frame_id to detections
            frame_size: (width, height) of frames
            
        Returns:
            Vanishing point coordinates (x, y)
        """
        print("🔍 Detecting vanishing point from vehicle trajectories...")
        
        # Extract motion vectors from consecutive detections
        motion_vectors = []
        frame_ids = sorted(detections_by_frame.keys())
        
        for i in range(len(frame_ids) - 1):
            curr_frame = frame_ids[i]
            next_frame = frame_ids[i + 1]
            
            curr_detections = detections_by_frame[curr_frame]
            next_detections = detections_by_frame[next_frame]
            
            # Simple nearest neighbor matching
            for curr_det in curr_detections:
                # Find closest detection in next frame
                min_dist = float('inf')
                best_match = None
                
                for next_det in next_detections:
                    if next_det.class_name == curr_det.class_name:
                        dist = np.sqrt((next_det.center_base[0] - curr_det.center_base[0])**2 + 
                                     (next_det.center_base[1] - curr_det.center_base[1])**2)
                        if dist < min_dist and dist > 2:  # Minimum movement threshold
                            min_dist = dist
                            best_match = next_det
                
                if best_match and min_dist < 100:  # Maximum movement threshold
                    # Create motion vector
                    dx = best_match.center_base[0] - curr_det.center_base[0]
                    dy = best_match.center_base[1] - curr_det.center_base[1]
                    
                    motion_vectors.append({
                        'start': curr_det.center_base,
                        'end': best_match.center_base,
                        'direction': (dx, dy)
                    })
        
        # Find vanishing point using RANSAC on motion lines
        if len(motion_vectors) < 5:
            print("⚠️ Not enough motion vectors, using default VP")
            return (frame_size[0] // 2, frame_size[1] // 3)
        
        vp = self._ransac_vanishing_point(motion_vectors, frame_size)
        print(f"✅ Detected vanishing point: {vp}")
        
        return vp
    
    def _ransac_vanishing_point(self, motion_vectors: List[Dict], 
                                frame_size: Tuple[int, int], 
                                iterations: int = 100) -> Tuple[int, int]:
        """
        RANSAC-based vanishing point estimation
        """
        best_vp = None
        best_inliers = 0
        w, h = frame_size
        
        for _ in range(iterations):
            if len(motion_vectors) < 2:
                continue
            
            # Sample two random motion vectors
            indices = np.random.choice(len(motion_vectors), 2, replace=False)
            v1 = motion_vectors[indices[0]]
            v2 = motion_vectors[indices[1]]
            
            # Extend lines and find intersection
            line1 = (*v1['start'], v1['end'][0], v1['end'][1])
            line2 = (*v2['start'], v2['end'][0], v2['end'][1])
            
            vp_candidate = self._line_intersection(line1, line2)
            
            if vp_candidate is None:
                continue
            
            # Check if VP is within reasonable bounds
            if not (-w < vp_candidate[0] < 2*w and -h < vp_candidate[1] < 2*h):
                continue
            
            # Count inliers
            inliers = 0
            for v in motion_vectors:
                line = (*v['start'], v['end'][0], v['end'][1])
                dist = self._point_to_line_distance(vp_candidate, line)
                if dist < 15:  # Threshold in pixels
                    inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_vp = vp_candidate
        
        # Default if no good VP found
        if best_vp is None:
            best_vp = (w // 2, h // 3)
        
        return best_vp
    
    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Optional[Tuple[int, int]]:
        """Find intersection of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        
        return (int(x), int(y))
    
    def _point_to_line_distance(self, point: Tuple[int, int], line: Tuple) -> float:
        """Calculate distance from point to line"""
        px, py = point
        x1, y1, x2, y2 = line
        
        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2
        
        dist = abs(A*px + B*py + C) / np.sqrt(A**2 + B**2 + 1e-10)
        
        return dist


class RFDETRCalibrationSystem:
    """
    Main calibration system using RF-DETR for vehicle detection
    Implements the automatic calibration approach from Dubská et al.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the calibration system
        
        Args:
            config: Configuration dictionary (merged with defaults)
        """
        # Start with default config and merge custom config
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Initialize components
        self.detector = RFDETRVehicleDetector(
            confidence_threshold=self.config['confidence_threshold']
        )
        self.vp_detector = VanishingPointDetector()
        
        # Calibration results
        self.vp1 = None  # First vanishing point (traffic direction)
        self.vp2 = None  # Second vanishing point (perpendicular to road)
        self.vp3 = None  # Third vanishing point (vertical)
        self.scale_factor = None  # Pixels per meter
        self.calibration_matrix = None
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'confidence_threshold': 0.35,
            'grid_spacing': 3.0,  # meters
            'lane_width': 3.5,    # meters  
            'max_calibration_frames': 300,
            'vehicle_dimensions': {  # Average dimensions in meters
                'car': {'length': 4.5, 'width': 1.8, 'height': 1.5},
                'bus': {'length': 11.0, 'width': 2.5, 'height': 3.2},
                'truck': {'length': 7.5, 'width': 2.3, 'height': 2.8},
                'motorcycle': {'length': 2.2, 'width': 0.8, 'height': 1.2}
            }
        }
    
    def calibrate_from_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Perform automatic calibration using RF-DETR detections
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save calibration results
            
        Returns:
            Dictionary containing calibration parameters
        """
        print("\n" + "="*70)
        print("🎯 RF-DETR AUTOMATIC CAMERA CALIBRATION SYSTEM")
        print("   Based on Dubská et al. approach with RF-DETR detection")
        print("="*70 + "\n")
        
        # Step 1: Collect vehicle detections
        print("Step 1: Collecting vehicle detections with RF-DETR...")
        detections_by_frame = self._collect_detections(video_path)
        
        # Step 2: Detect vanishing point from vehicle motion
        print("\nStep 2: Detecting vanishing point...")
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        self.vp1 = self.vp_detector.detect_from_vehicle_tracks(
            detections_by_frame, (frame_width, frame_height)
        )
        
        # Step 3: Estimate scale factor
        print("\nStep 3: Estimating scale factor...")
        self.scale_factor = self._estimate_scale_factor(detections_by_frame)
        
        # Step 4: Generate calibration data
        print("\nStep 4: Generating calibration grid...")
        calibration_data = self._generate_calibration_data(frame_width, frame_height)
        
        # Save calibration if requested
        if output_path:
            self._save_calibration(calibration_data, output_path)
        
        print("\n" + "="*70)
        print("✅ CALIBRATION COMPLETE!")
        print(f"   Vanishing Point: {self.vp1}")
        print(f"   Scale Factor: {self.scale_factor:.2f} pixels/meter")
        print(f"   Total Vehicles Detected: {sum(len(d) for d in detections_by_frame.values())}")
        print("="*70 + "\n")
        
        return calibration_data
    
    def _collect_detections(self, video_path: str) -> Dict[int, List[VehicleDetection]]:
        """
        Collect vehicle detections from video
        """
        cap = cv2.VideoCapture(video_path)
        
        detections_by_frame = {}
        frame_id = 0
        max_frames = self.config['max_calibration_frames']
        
        print(f"Processing up to {max_frames} frames...")
        
        while frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles with RF-DETR
            detections = self.detector.detect_vehicles(frame, frame_id)
            
            # Apply NMS to remove duplicates
            detections = self.detector.apply_nms(detections)
            
            if detections:
                detections_by_frame[frame_id] = detections
            
            # Progress update
            if frame_id % 50 == 0:
                total_detections = sum(len(d) for d in detections_by_frame.values())
                print(f"   Frame {frame_id}: {total_detections} total vehicles detected")
            
            frame_id += 1
        
        cap.release()
        
        total = sum(len(d) for d in detections_by_frame.values())
        print(f"✅ Detected {total} vehicles across {len(detections_by_frame)} frames")
        
        return detections_by_frame
    
    def _estimate_scale_factor(self, detections_by_frame: Dict[int, List[VehicleDetection]]) -> float:
        """
        Estimate scale factor using vehicle dimensions
        """
        all_detections = []
        for detections in detections_by_frame.values():
            all_detections.extend(detections)
        
        if not all_detections:
            print("⚠️ No detections available, using default scale")
            return 50.0  # Default pixels per meter
        
        scales = []
        
        for det in all_detections:
            if det.class_name in self.config['vehicle_dimensions']:
                expected_dims = self.config['vehicle_dimensions'][det.class_name]
                
                # Calculate scale from width (more reliable than height)
                bbox_width = det.bbox[2] - det.bbox[0]
                scale = bbox_width / expected_dims['width']
                
                # Filter outliers
                if 20 < scale < 200:  # Reasonable range for pixels/meter
                    scales.append(scale)
        
        if scales:
            # Use median for robustness
            scale_factor = np.median(scales)
            print(f"   Estimated from {len(scales)} measurements")
        else:
            scale_factor = 50.0  # Default
            print("   Using default scale factor")
        
        return scale_factor
    
    def _generate_calibration_data(self, frame_width: int, frame_height: int) -> Dict:
        """
        Generate calibration data including grid points
        """
        grid_points = []
        
        if self.vp1:
            # Generate perspective grid based on vanishing point
            grid_spacing_px = self.scale_factor * self.config['grid_spacing']
            
            for i in range(-5, 6):  # Horizontal lines
                for j in range(0, 10):  # Depth lines
                    # Apply perspective transformation
                    depth_factor = 1.0 / (j + 1)
                    x = self.vp1[0] + i * grid_spacing_px * depth_factor
                    y = self.vp1[1] + j * grid_spacing_px * 0.5
                    
                    if 0 <= x < frame_width and 0 <= y < frame_height:
                        grid_points.append({
                            'pixel': (int(x), int(y)),
                            'world': (i * self.config['grid_spacing'], 
                                    j * self.config['grid_spacing'], 0)
                        })
        
        calibration_data = {
            'vp1': self.vp1,
            'scale_factor': self.scale_factor,
            'grid_points': grid_points,
            'grid_spacing_m': self.config['grid_spacing'],
            'frame_size': (frame_width, frame_height),
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'model': 'RF-DETR',
            'confidence_threshold': self.config['confidence_threshold']
        }
        
        return calibration_data
    
    def _save_calibration(self, calibration_data: Dict, output_path: str):
        """Save calibration data to JSON file"""
        # Convert to JSON-serializable format
        json_data = {
            'vp1': list(calibration_data['vp1']) if calibration_data['vp1'] else None,
            'scale_factor': float(calibration_data['scale_factor']),
            'grid_spacing_m': calibration_data['grid_spacing_m'],
            'frame_size': list(calibration_data['frame_size']),
            'timestamp': calibration_data['timestamp'],
            'model': calibration_data['model'],
            'confidence_threshold': calibration_data['confidence_threshold'],
            'grid_points': [
                {
                    'pixel': list(p['pixel']),
                    'world': list(p['world'])
                }
                for p in calibration_data['grid_points'][:50]  # Save subset
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"💾 Calibration saved to: {output_path}")
    
    def visualize_calibration(self, video_path: str, calibration_data: Dict, 
                             output_video: Optional[str] = None, display: bool = False):
        """
        Visualize calibration results with RF-DETR detections
        """
        print("\n🎨 Visualizing calibration with RF-DETR detections...")
        
        cap = cv2.VideoCapture(video_path)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frame_id = 0
        max_vis_frames = 150
        
        while frame_id < max_vis_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw vanishing point
            if calibration_data['vp1']:
                cv2.circle(frame, calibration_data['vp1'], 12, (0, 255, 0), -1)
                cv2.putText(frame, "VP1", 
                          (calibration_data['vp1'][0] + 15, calibration_data['vp1'][1]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw grid
            for point in calibration_data['grid_points']:
                cv2.circle(frame, point['pixel'], 3, (255, 255, 0), -1)
            
            # Detect and visualize vehicles
            detections = self.detector.detect_vehicles(frame, frame_id)
            
            for det in detections:
                # Draw bounding box
                color = (0, 255, 0) if det.class_name == 'car' else (255, 100, 0)
                cv2.rectangle(frame, (det.bbox[0], det.bbox[1]), 
                            (det.bbox[2], det.bbox[3]), color, 2)
                
                # Draw label
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(frame, label, (det.bbox[0], det.bbox[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw estimated base corners
                for corner_name, corner_point in det.corners.items():
                    if corner_name not in ['bbox', 'top_left', 'top_right']:
                        cv2.circle(frame, corner_point, 4, (0, 0, 255), -1)
            
            # Add info overlay
            info = f"RF-DETR | Frame: {frame_id} | Scale: {calibration_data['scale_factor']:.1f} px/m"
            cv2.putText(frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if output_video:
                out.write(frame)
            
            if display:
                cv2.imshow('RF-DETR Calibration', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_id += 1
        
        cap.release()
        if output_video:
            out.release()
            print(f"✅ Visualization saved to: {output_video}")
        
        cv2.destroyAllWindows()


def main():
    """
    Main demonstration of RF-DETR based calibration
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         RF-DETR AUTOMATIC CAMERA CALIBRATION SYSTEM          ║
    ║                                                               ║
    ║  Roboflow DETR + Automatic Calibration (Dubská et al.)      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Example usage
    video_path = "video/Traffic_3.mp4"  # Replace with your video
    
    # Configuration
    config = {
        'confidence_threshold': 0.35,
        'grid_spacing': 3.0,
        'max_calibration_frames': 300
    }
    
    try:
        # Initialize system
        calibrator = RFDETRCalibrationSystem(config)
        
        # Perform calibration
        calibration_data = calibrator.calibrate_from_video(
            video_path,
            output_path="rfdetr_calibration.json"
        )
        
        # Visualize results
        calibrator.visualize_calibration(
            video_path,
            calibration_data,
            output_video="rfdetr_calibration_viz.mp4",
            display=False
        )
        
        print("\n✨ Ready for speed and distance measurements!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
    Installation Instructions:
    ==========================
    
    1. Install RF-DETR dependencies:
       pip install inference-gpu  # For GPU support
       OR
       pip install inference      # For CPU only
    
    2. Install additional requirements:
       pip install opencv-python numpy torch
       pip install roboflow supervision
    
    3. Get a free Roboflow API key (optional):
       - Sign up at https://app.roboflow.com
       - Or use local inference without API key
    
    4. Update 'video_path' in main() with your video
    
    5. Run: python rfdetr_calibration_implementation.py
    
    RF-DETR Advantages:
    - Optimized for production environments
    - Better small object detection
    - Active development and support
    - Works well with COCO pretrained weights
    """)
    
    # Uncomment to run
    main()
