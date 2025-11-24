import cv2
import numpy as np
import json
import os

class CalibrationValidator:
    def __init__(self, video_path, calibration_file):
        self.video_path = video_path
        self.calibration_file = calibration_file
        
        # State for drawing
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_frame = None
        self.validation_result = None
        
        # Load calibration data
        self.load_calibration()

    def load_calibration(self):
        """Load the saved JSON calibration grid"""
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"Cannot find {self.calibration_file}. Run calibration first!")
            
        with open(self.calibration_file, 'r') as f:
            data = json.load(f)
            
        self.num_lanes = data['num_lanes']
        self.num_depth_zones = data['num_depth_zones']
        
        # Load grid and Vanishing Point
        self.grid = data['grid'] 
        self.vp1 = data['vp1']  # Store VP1 directly
        print(f"✅ Loaded calibration grid with {len(self.grid)} zones")

    def get_ppm_at_point(self, x, y, img_width, img_height):
        """
        Look up the PPM for a specific (x,y) location.
        UPDATED: Uses Vanishing Point logic to match the tilted visualization.
        """
        vp_x, vp_y = self.vp1
        
        # --- 1. Calculate Lane Index (Perspective Logic) ---
        # Avoid division by zero if point is above VP
        if y <= vp_y + 1: 
            lane_idx = int(x / (img_width / self.num_lanes))
        else:
            # Project point to bottom of screen to find true lane
            t = (img_height - vp_y) / (y - vp_y)
            x_projected = vp_x + (x - vp_x) * t
            
            lane_width_at_bottom = img_width / self.num_lanes
            lane_idx = int(x_projected / lane_width_at_bottom)
            
        # --- 2. Calculate Depth Index ---
        depth_height = img_height / self.num_depth_zones
        depth_idx = int(y / depth_height)
        
        # Clamp indices
        lane_idx = np.clip(lane_idx, 0, self.num_lanes - 1)
        depth_idx = np.clip(depth_idx, 0, self.num_depth_zones - 1)
        
        # Lookup PPM
        key = f"{lane_idx}_{depth_idx}"
        return self.grid.get(key, 40)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.validation_result = None 

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.calculate_distance()

    def calculate_distance(self):
        if not self.start_point or not self.end_point:
            return

        x1, y1 = self.start_point
        x2, y2 = self.end_point
        img_h, img_w = self.current_frame.shape[:2]

        # 1. Pixel Distance
        dist_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # 2. Get PPM at midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Now this uses the CORRECT perspective logic
        ppm = self.get_ppm_at_point(mid_x, mid_y, img_w, img_h)
        
        # 3. Real Meters
        dist_m = dist_px / ppm
        
        self.validation_result = {
            'pixels': dist_px,
            'ppm': ppm,
            'meters': dist_m,
            'midpoint': (int(mid_x), int(mid_y))
        }

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Cannot read video")
            return
            
        self.current_frame = frame
        cv2.namedWindow("Validation Tool")
        cv2.setMouseCallback("Validation Tool", self.mouse_callback)

        print("\n" + "="*50)
        print("📏 CALIBRATION VALIDATION TOOL (TILTED VIEW)")
        print("="*50)
        print("1. Draw a line across a lane width.")
        print("2. 'n' for new frame, 'q' to quit.")

        while True:
            display_frame = self.current_frame.copy()
            h, w = display_frame.shape[:2]
            
            # --- Visualize Tilted Grid ---
            vp_x, vp_y = int(self.vp1[0]), int(self.vp1[1]) # Fixed access
            
            # Draw VP
            cv2.circle(display_frame, (vp_x, vp_y), 8, (0, 165, 255), -1)
            cv2.putText(display_frame, "VP", (vp_x+10, vp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # Draw Lane Dividers (Rays)
            for i in range(1, self.num_lanes):
                x_bottom = int(i * (w / self.num_lanes))
                cv2.line(display_frame, (vp_x, vp_y), (x_bottom, h), (50, 50, 50), 1)
            
            # Draw Depth Dividers
            for i in range(1, self.num_depth_zones):
                y = int(i * (h / self.num_depth_zones))
                cv2.line(display_frame, (0, y), (w, y), (50, 50, 50), 1)
            
            # Draw Measure Line
            if self.start_point and self.end_point:
                color = (0, 255, 255) if self.drawing else (0, 255, 0)
                cv2.line(display_frame, self.start_point, self.end_point, color, 2)
                
            # Display Result
            if self.validation_result:
                res = self.validation_result
                text = f"{res['meters']:.2f} m"
                pt = res['midpoint']
                cv2.putText(display_frame, text, (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                info = f"PPM: {res['ppm']:.1f} (Lane Corrected)"
                cv2.putText(display_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow("Validation Tool", display_frame)
            
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'): break
            elif key == ord('n'):
                for _ in range(30): cap.read()
                ret, f = cap.read()
                if ret: 
                    self.current_frame = f
                    self.start_point = None
                    self.validation_result = None

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_PATH = "video/Traffic_3.mp4" 
    CALIB_FILE = "hybrid_calibration_yolo11.json"
    
    try:
        validator = CalibrationValidator(VIDEO_PATH, CALIB_FILE)
        validator.run()
    except Exception as e:
        print(f"Error: {e}")