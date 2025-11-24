import cv2
import numpy as np

class SimplifiedVPDetector:
    """
    Simplified vanishing point detection
    - Uses motion tracking instead of complex diamond space
    - Good enough for automatic grid population
    """
    
    def __init__(self):
        self.motion_vectors = []
        
    def detect_traffic_direction(self, video_path, num_frames=150):
        """
        Detect dominant traffic direction from vehicle motion
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze (default: 150)
        
        Returns:
            vp1: First vanishing point (x, y) - traffic direction
        """
        print("🔍 Detecting traffic direction...")
        
        cap = cv2.VideoCapture(video_path)
        
        # Initialize optical flow tracker
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Read first frame
        ret, old_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read video")
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect initial features
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        frame_count = 0
        motion_vectors = []
        
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params
                )
                
                # Select good points
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    # Calculate motion vectors
                    for new, old in zip(good_new, good_old):
                        # Motion vector
                        dx = new[0] - old[0]
                        dy = new[1] - old[1]
                        
                        # Filter significant motion (> 2 pixels)
                        if np.sqrt(dx**2 + dy**2) > 2:
                            motion_vectors.append((dx, dy, old[0], old[1]))
            
            # Update for next iteration
            old_gray = frame_gray.copy()
            
            # Re-detect features every 10 frames
            if frame_count % 10 == 0:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            else:
                p0 = good_new.reshape(-1, 1, 2) if p1 is not None else None
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"   Processed {frame_count}/{num_frames} frames...")
        
        cap.release()
        
        print(f"✅ Collected {len(motion_vectors)} motion vectors")
        
        # Compute vanishing point from motion vectors
        vp1 = self._compute_vp_from_motion(motion_vectors, old_frame.shape)
        
        return vp1
    
    def _compute_vp_from_motion(self, motion_vectors, frame_shape):
        """
        Compute vanishing point from motion vectors
        Uses line intersection voting
        """
        if len(motion_vectors) < 10:
            # Fallback: assume VP at center-top
            height, width = frame_shape[:2]
            return (width // 2, 0)
        
        height, width = frame_shape[:2]
        
        # Create lines from motion vectors
        # Each line: point + direction → extends to find intersection
        lines = []
        for dx, dy, x, y in motion_vectors:
            # Normalize direction
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                continue
            
            dx_norm = dx / length
            dy_norm = dy / length
            
            lines.append({
                'point': (x, y),
                'direction': (dx_norm, dy_norm)
            })
        
        # Vote for vanishing point using line intersections
        # Accumulate intersection points
        intersection_points = []
        
        # Sample pairs of lines
        sample_size = min(len(lines), 200)
        sampled_indices = np.random.choice(len(lines), sample_size, replace=False)
        
        for i in range(sample_size):
            for j in range(i + 1, min(i + 10, sample_size)):  # Compare with next 10
                idx1, idx2 = sampled_indices[i], sampled_indices[j]
                line1 = lines[idx1]
                line2 = lines[idx2]
                
                # Find intersection
                intersection = self._line_intersection(line1, line2, frame_shape)
                if intersection is not None:
                    intersection_points.append(intersection)
        
        if len(intersection_points) == 0:
            # Fallback
            return (width // 2, 0)
        
        # Cluster intersection points and find dominant cluster
        intersection_points = np.array(intersection_points)
        
        # Use median (robust to outliers)
        vp_x = np.median(intersection_points[:, 0])
        vp_y = np.median(intersection_points[:, 1])
        
        # Clamp to reasonable range
        vp_x = np.clip(vp_x, -width, width * 2)
        vp_y = np.clip(vp_y, -height, height * 2)
        
        print(f"✅ Detected VP1: ({vp_x:.1f}, {vp_y:.1f})")
        
        return (vp_x, vp_y)
    
    def _line_intersection(self, line1, line2, frame_shape):
        """
        Find intersection of two lines defined by point + direction
        """
        p1, d1 = line1['point'], line1['direction']
        p2, d2 = line2['point'], line2['direction']
        
        # Line 1: p1 + t1 * d1
        # Line 2: p2 + t2 * d2
        # Solve: p1 + t1*d1 = p2 + t2*d2
        
        # Matrix form: [d1x -d2x] [t1] = [p2x - p1x]
        #              [d1y -d2y] [t2]   [p2y - p1y]
        
        A = np.array([[d1[0], -d2[0]], 
                      [d1[1], -d2[1]]])
        b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        
        # Check if lines are parallel
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return None
        
        # Solve
        try:
            t = np.linalg.solve(A, b)
            t1 = t[0]
            
            # Compute intersection point
            intersection = (p1[0] + t1 * d1[0], p1[1] + t1 * d1[1])
            
            # Filter unreasonable intersections
            height, width = frame_shape[:2]
            x, y = intersection
            
            # Accept only if within reasonable range
            if -width < x < width * 2 and -height < y < height * 2:
                return intersection
            
        except np.linalg.LinAlgError:
            pass
        
        return None
