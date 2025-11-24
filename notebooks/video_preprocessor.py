"""
Video Preprocessor for Traffic Detection
Provides adaptive preprocessing techniques for various traffic scenarios
to improve object detection performance.
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict, Any


class ScenarioType(Enum):
    """Enumeration of different traffic video scenarios"""
    DAYTIME = "daytime"
    NIGHTTIME = "nighttime"
    RAINY_FOGGY = "rainy_foggy"
    HIGH_SPEED = "high_speed"
    MIXED_LIGHTING = "mixed_lighting"


class ComprehensiveVideoPreprocessor:
    """
    Comprehensive video preprocessor that adapts processing techniques
    based on detected scenario conditions.
    
    Attributes:
        auto_detect_scenario (bool): Whether to automatically detect scenario
        manual_scenario (ScenarioType): Manual scenario override
        prev_frame: Previous frame for motion analysis
        frame_buffer: Buffer for frame averaging
        buffer_size (int): Size of frame buffer for motion blur reduction
    """
    
    def __init__(self, 
                 auto_detect_scenario: bool = True, 
                 manual_scenario: Optional[ScenarioType] = None,
                 buffer_size: int = 3):
        """
        Initialize the video preprocessor.
        
        Args:
            auto_detect_scenario: Whether to automatically detect scenario
            manual_scenario: Manual scenario override (used if auto_detect is False)
            buffer_size: Size of frame buffer for motion blur reduction
        """
        self.auto_detect_scenario = auto_detect_scenario
        self.manual_scenario = manual_scenario
        self.prev_frame = None
        self.frame_buffer = []
        self.buffer_size = buffer_size
        
    def analyze_frame_conditions(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frame to determine scenario and parameters.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing scenario and analysis metrics
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        contrast = gray.std()
        
        # Detect edges for motion analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Analyze color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_mean = np.mean(hsv[:,:,1])
        
        # Determine scenario based on metrics
        if mean_brightness < 80:
            scenario = ScenarioType.NIGHTTIME
        elif contrast < 25:
            scenario = ScenarioType.RAINY_FOGGY
        elif edge_density > 0.15:  # High edge density might indicate motion
            scenario = ScenarioType.HIGH_SPEED
        elif contrast > 50 and saturation_mean < 100:
            scenario = ScenarioType.MIXED_LIGHTING
        else:
            scenario = ScenarioType.DAYTIME
            
        return {
            'scenario': scenario,
            'mean_brightness': mean_brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'saturation': saturation_mean
        }
    
    def enhance_illumination_adaptive(self, 
                                     frame: np.ndarray, 
                                     conditions: Dict[str, Any]) -> np.ndarray:
        """
        Adaptive illumination enhancement based on conditions.
        
        Args:
            frame: Input frame
            conditions: Dictionary with scenario and metrics
            
        Returns:
            Enhanced frame
        """
        if conditions['scenario'] == ScenarioType.NIGHTTIME:
            # Strong gamma correction for night
            frame = self.adjust_gamma(frame, gamma=1.8)
            
        # CLAHE with adaptive parameters
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adjust CLAHE parameters based on scenario
        if conditions['scenario'] == ScenarioType.NIGHTTIME:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
        elif conditions['scenario'] == ScenarioType.RAINY_FOGGY:
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        elif conditions['scenario'] == ScenarioType.MIXED_LIGHTING:
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(12,12))
        else:  # DAYTIME
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def adjust_gamma(self, frame: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Apply gamma correction to adjust brightness.
        
        Args:
            frame: Input frame
            gamma: Gamma value (>1 brightens, <1 darkens)
            
        Returns:
            Gamma corrected frame
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)
    
    def denoise_adaptive(self, 
                        frame: np.ndarray, 
                        conditions: Dict[str, Any]) -> np.ndarray:
        """
        Adaptive denoising based on conditions.
        
        Args:
            frame: Input frame
            conditions: Dictionary with scenario and metrics
            
        Returns:
            Denoised frame
        """
        if conditions['scenario'] == ScenarioType.NIGHTTIME:
            # Stronger denoising for night (more noise expected)
            return cv2.bilateralFilter(frame, d=9, sigmaColor=100, sigmaSpace=100)
        elif conditions['scenario'] == ScenarioType.HIGH_SPEED:
            # Lighter denoising to preserve motion details
            return cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
        else:
            # Standard denoising
            return cv2.bilateralFilter(frame, d=7, sigmaColor=75, sigmaSpace=75)
    
    def sharpen_adaptive(self, 
                        frame: np.ndarray, 
                        conditions: Dict[str, Any]) -> np.ndarray:
        """
        Adaptive sharpening based on scenario.
        
        Args:
            frame: Input frame
            conditions: Dictionary with scenario and metrics
            
        Returns:
            Sharpened frame
        """
        if conditions['scenario'] == ScenarioType.DAYTIME:
            # Mild sharpening for daytime
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            return cv2.filter2D(frame, -1, kernel)
        elif conditions['scenario'] != ScenarioType.HIGH_SPEED:
            # Unsharp masking for other scenarios (except high-speed)
            gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
            return cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        return frame
    
    def dehaze(self, frame: np.ndarray) -> np.ndarray:
        """
        Dehazing for foggy/rainy conditions using dark channel prior.
        
        Args:
            frame: Input frame
            
        Returns:
            Dehazed frame
        """
        # Dark channel prior
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = np.min(frame, axis=2)
        dark_channel = cv2.morphologyEx(dark_channel, cv2.MORPH_CLOSE, kernel)
        
        # Atmospheric light estimation
        flat = dark_channel.flatten()
        num_pixels = len(flat)
        top_pixels = max(int(num_pixels * 0.001), 1)
        indices = np.argpartition(flat, -top_pixels)[-top_pixels:]
        
        atmospheric_light = np.max(frame.reshape(-1, 3)[indices], axis=0)
        atmospheric_light = np.maximum(atmospheric_light, 1)  # Avoid division by zero
        
        # Transmission map
        transmission = 1 - 0.95 * (dark_channel / np.max(atmospheric_light))
        transmission = np.maximum(transmission, 0.1)
        
        # Recover scene
        dehazed = np.zeros_like(frame, dtype=np.float32)
        for c in range(3):
            dehazed[:,:,c] = ((frame[:,:,c].astype(np.float32) - atmospheric_light[c]) / 
                             transmission + atmospheric_light[c])
        
        return np.clip(dehazed, 0, 255).astype(np.uint8)
    
    def reduce_motion_blur(self, frame: np.ndarray) -> np.ndarray:
        """
        Motion blur reduction using frame averaging.
        
        Args:
            frame: Input frame
            
        Returns:
            Motion-reduced frame
        """
        self.frame_buffer.append(frame.copy())
        
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) >= 2:
            # Weighted averaging with emphasis on current frame
            weights = np.array([0.2, 0.3, 0.5])[-len(self.frame_buffer):]
            weights = weights / weights.sum()
            
            result = np.zeros_like(frame, dtype=np.float32)
            for i, f in enumerate(self.frame_buffer):
                result += f.astype(np.float32) * weights[i]
            
            return result.astype(np.uint8)
        return frame
    
    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple frame stabilization for high-speed scenarios.
        
        Args:
            frame: Input frame
            
        Returns:
            Stabilized frame
        """
        if self.prev_frame is not None:
            try:
                # Detect features
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
                
                # Find good features to track
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, 
                                                  qualityLevel=0.01, minDistance=30)
                
                if prev_pts is not None:
                    # Calculate optical flow
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                    
                    # Filter good points
                    prev_pts = prev_pts[status == 1]
                    next_pts = next_pts[status == 1]
                    
                    if len(prev_pts) >= 4:
                        # Estimate transform
                        transform, _ = cv2.estimateAffinePartial2D(prev_pts, next_pts)
                        
                        if transform is not None:
                            # Apply stabilization (mild correction)
                            dx = transform[0, 2] * 0.1  # Reduce correction strength
                            dy = transform[1, 2] * 0.1
                            
                            stabilization_matrix = np.array([[1, 0, -dx],
                                                            [0, 1, -dy]], dtype=np.float32)
                            
                            frame = cv2.warpAffine(frame, stabilization_matrix, 
                                                 (frame.shape[1], frame.shape[0]))
            except Exception as e:
                # If stabilization fails, return original frame
                print(f"Stabilization failed: {e}")
        
        return frame
    
    def enhance_local_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Local contrast enhancement for mixed lighting.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply adaptive histogram equalization to L channel
        clahe_local = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        l = clahe_local.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Additional local contrast using bilateral grid
        return cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    
    def optimize_color_space(self, 
                           frame: np.ndarray, 
                           conditions: Dict[str, Any]) -> np.ndarray:
        """
        Color space optimization based on conditions.
        
        Args:
            frame: Input frame
            conditions: Dictionary with scenario and metrics
            
        Returns:
            Color-optimized frame
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        if conditions['scenario'] == ScenarioType.NIGHTTIME:
            # Reduce saturation slightly for night
            hsv[:,:,1] = hsv[:,:,1] * 0.9
            hsv[:,:,2] = np.minimum(hsv[:,:,2] * 1.2, 255)  # Increase value
        elif conditions['scenario'] == ScenarioType.DAYTIME:
            # Enhance colors for daytime
            hsv[:,:,1] = np.minimum(hsv[:,:,1] * 1.2, 255)
        elif conditions['scenario'] == ScenarioType.RAINY_FOGGY:
            # Increase saturation for foggy conditions
            hsv[:,:,1] = np.minimum(hsv[:,:,1] * 1.4, 255)
            
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, ScenarioType]:
        """
        Main processing pipeline that adapts to conditions.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, detected_scenario)
        """
        processed = frame.copy()
        
        # Determine scenario
        if self.auto_detect_scenario:
            conditions = self.analyze_frame_conditions(frame)
        else:
            conditions = {'scenario': self.manual_scenario}
        
        scenario = conditions['scenario']
        
        # Apply scenario-specific processing pipeline
        if scenario == ScenarioType.DAYTIME:
            # Daytime: CLAHE + mild sharpening
            processed = self.enhance_illumination_adaptive(processed, conditions)
            processed = self.sharpen_adaptive(processed, conditions)
            
        elif scenario == ScenarioType.NIGHTTIME:
            # Night: Gamma correction + strong noise reduction + CLAHE
            processed = self.adjust_gamma(processed, gamma=1.8)
            processed = self.denoise_adaptive(processed, conditions)
            processed = self.enhance_illumination_adaptive(processed, conditions)
            processed = self.optimize_color_space(processed, conditions)
            
        elif scenario == ScenarioType.RAINY_FOGGY:
            # Rainy/Foggy: Dehazing + contrast enhancement
            processed = self.dehaze(processed)
            processed = self.enhance_illumination_adaptive(processed, conditions)
            processed = self.denoise_adaptive(processed, conditions)
            processed = self.optimize_color_space(processed, conditions)
            
        elif scenario == ScenarioType.HIGH_SPEED:
            # High-speed: Motion blur reduction + frame stabilization
            processed = self.stabilize_frame(processed)
            processed = self.reduce_motion_blur(processed)
            processed = self.denoise_adaptive(processed, conditions)
            processed = self.enhance_illumination_adaptive(processed, conditions)
            
        elif scenario == ScenarioType.MIXED_LIGHTING:
            # Mixed lighting: Adaptive histogram + local contrast enhancement
            processed = self.enhance_illumination_adaptive(processed, conditions)
            processed = self.enhance_local_contrast(processed)
            processed = self.optimize_color_space(processed, conditions)
        
        self.prev_frame = frame.copy()
        return processed, scenario
    
    def reset(self):
        """Reset the preprocessor state (useful between different videos)."""
        self.prev_frame = None
        self.frame_buffer = []


# Convenience function for quick setup
def create_preprocessor(scenario: Optional[str] = None, 
                       auto_detect: bool = True) -> ComprehensiveVideoPreprocessor:
    """
    Create a video preprocessor with specified settings.
    
    Args:
        scenario: Optional scenario name ('daytime', 'nighttime', etc.)
        auto_detect: Whether to auto-detect scenario
        
    Returns:
        Configured ComprehensiveVideoPreprocessor instance
    """
    manual_scenario = None
    if scenario and not auto_detect:
        scenario_map = {
            'daytime': ScenarioType.DAYTIME,
            'nighttime': ScenarioType.NIGHTTIME,
            'rainy_foggy': ScenarioType.RAINY_FOGGY,
            'high_speed': ScenarioType.HIGH_SPEED,
            'mixed_lighting': ScenarioType.MIXED_LIGHTING
        }
        manual_scenario = scenario_map.get(scenario)
    
    return ComprehensiveVideoPreprocessor(
        auto_detect_scenario=auto_detect,
        manual_scenario=manual_scenario
    )


if __name__ == "__main__":
    # Example usage
    print(f"Available scenarios: {[s.value for s in ScenarioType]}")
    
    # Create preprocessor with auto-detection
    preprocessor = create_preprocessor(auto_detect=True)
    print("Preprocessor created with auto-detection enabled")
