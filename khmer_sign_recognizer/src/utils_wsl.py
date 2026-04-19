"""
WSL Layer - Utility Functions
"""

import json
import logging
import numpy as np
from typing import Dict, Optional

def load_config(config_path: str = "config/settings.json") -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)

# ============================================================================
# Math Utilities
# ============================================================================

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(p1 - p2)

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def vector_to_quaternion(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    v_from = normalize_vector(v_from)
    v_to = normalize_vector(v_to)
    
    dot = np.dot(v_from, v_to)
    dot = np.clip(dot, -1.0, 1.0)
    
    # If vectors are nearly identical, return no rotation
    if dot > 0.9999:
        return np.array([0, 0, 0, 1])
    
    # If vectors are nearly opposite, find perpendicular axis
    if dot < -0.9999:
        perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
        axis = normalize_vector(np.cross(v_from, perp))
        return np.array([axis[0], axis[1], axis[2], 0])
    
    axis = np.cross(v_from, v_to)
    angle = np.arccos(dot)
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    
    axis = normalize_vector(axis)
    return np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        np.cos(half_angle)
    ])

# ============================================================================
# One Euro Filter for Smoothing
# ============================================================================

class OneEuroFilter:
    """One Euro Filter for smooth landmark tracking"""
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def __call__(self, x, t):
        """Apply filter to value x at time t"""
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        # Calculate time delta
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        
        # Calculate derivative
        dx = (x - self.x_prev) / dt
        
        # Smooth derivative
        edx = self._smoothing_factor(self.d_cutoff, dt)
        dx_smoothed = self._exponential_smoothing(dx, self.dx_prev, edx)
        
        # Calculate cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_smoothed)
        
        # Smooth value
        ex = self._smoothing_factor(cutoff, dt)
        x_smoothed = self._exponential_smoothing(x, self.x_prev, ex)
        
        # Store for next iteration
        self.x_prev = x_smoothed
        self.dx_prev = dx_smoothed
        self.t_prev = t
        
        return x_smoothed
    
    def _smoothing_factor(self, cutoff, dt):
        """Calculate smoothing factor"""
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)
    
    def _exponential_smoothing(self, x, x_prev, alpha):
        """Exponential smoothing"""
        return alpha * x + (1 - alpha) * x_prev

# ============================================================================
# AI Framework Placeholder
# ============================================================================

class AIFramework:
    """Placeholder for future AI integration"""
    
    def __init__(self, config: Dict):
        self.enabled = config.get('ai_framework', {}).get('enabled', False)
        self.buffer = []
        self.buffer_size = config.get('ai_framework', {}).get('buffer_size', 60)
        self.model = None
        
        if self.enabled:
            logging.info("AI Framework enabled - ready for future model integration")
    
    def add_frame(self, bone_data: Dict):
        """Add frame to buffer"""
        if not self.enabled:
            return
        
        self.buffer.append(bone_data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def predict(self) -> Optional[str]:
        """Predict sign from buffer (placeholder)"""
        if not self.enabled or len(self.buffer) < self.buffer_size:
            return None
        
        # TODO: Implement actual prediction when model is ready
        # Example future code:
        # import torch
        # prediction = self.model(self.buffer)
        # return prediction
        
        return None
    
    def get_buffer_status(self) -> Dict:
        """Get buffer status"""
        return {
            'enabled': self.enabled,
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.buffer_size,
            'ready_for_prediction': len(self.buffer) >= self.buffer_size
        }
