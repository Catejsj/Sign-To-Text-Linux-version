"""
WSL Layer - Normalization and Bone Mapping
Updated to handle RTMPose named keypoint format
"""

import numpy as np
import logging
from typing import Dict, Optional
from src.utils_wsl import (
    calculate_distance,
    normalize_vector,
    vector_to_quaternion,
    OneEuroFilter
)

logger = logging.getLogger(__name__)


class SkeletonMapper:

    def __init__(self, config: Dict):
        self.config = config
        self.norm_config   = config['normalization']
        self.smooth_config = config['smoothing']

        self.filters = {}
        self.smoothing_enabled = self.smooth_config['enabled']

        LANDMARK_NAMES = [
            'nose', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_index', 'right_index',
        ]

        if self.smoothing_enabled:
            min_cutoff = self.smooth_config['min_cutoff']
            beta       = self.smooth_config['beta']
            for name in LANDMARK_NAMES:
                for axis in ('x', 'y', 'z'):
                    self.filters[f'{name}_{axis}'] = OneEuroFilter(min_cutoff, beta)

        self.processed_count = 0
        self.failed_count    = 0

        logger.info("SkeletonMapper initialized (RTMPose format)")
        logger.info(f"Smoothing: {'Enabled' if self.smoothing_enabled else 'Disabled'}")

    def process(self, landmark_data: Dict) -> Optional[Dict]:
        try:
            pose      = landmark_data.get('pose', {})
            left_hand = landmark_data.get('left_hand', {})
            right_hand = landmark_data.get('right_hand', {})
            timestamp = landmark_data.get('timestamp', 0)

            if not pose or len(pose) < 4:
                self.failed_count += 1
                return None

            # Convert pose dict to numpy points
            pose_points = self._extract_pose_points(pose, landmark_data)
            norm_params = self._calculate_normalization(pose_points)

            if norm_params is None:
                self.failed_count += 1
                return None

            normalized_pose = self._normalize_landmarks(
                pose_points, norm_params, timestamp
            )
            bone_transforms   = self._calculate_bone_transforms(normalized_pose)
            finger_transforms = {}

            if left_hand:
                finger_transforms.update(
                    self._calculate_finger_transforms(left_hand, 'left')
                )
            if right_hand:
                finger_transforms.update(
                    self._calculate_finger_transforms(right_hand, 'right')
                )

            self.processed_count += 1

            return {
                'frame_id':          landmark_data.get('frame_id', 0),
                'timestamp':         timestamp,
                'normalized_pose':   {k: v.tolist() for k, v in normalized_pose.items()},
                'bone_transforms':   bone_transforms,
                'finger_transforms': finger_transforms,
                'normalization_params': {
                    'anchor': norm_params['anchor'].tolist(),
                    'scale':  float(norm_params['scale'])
                }
            }

        except Exception as e:
            logger.error(f"Error processing landmarks: {e}")
            self.failed_count += 1
            return None

    def _extract_pose_points(self, pose: Dict,
                             landmark_data: Dict) -> Dict[str, np.ndarray]:
        """
        RTMPose gives pixel coordinates directly
        Convert to normalized 0-1 range using image dimensions
        """
        points = {}
        img_w = landmark_data.get('image_width',  640)
        img_h = landmark_data.get('image_height', 480)

        for name, lm in pose.items():
            # RTMPose outputs pixel coords - normalize to 0-1
            x = float(lm['x']) / img_w
            y = float(lm['y']) / img_h
            z = float(lm.get('z', 0.0))
            points[name] = np.array([x, y, z])

        return points

    def _calculate_normalization(self, pose_points: Dict) -> Optional[Dict]:
        if 'left_shoulder' not in pose_points or \
           'right_shoulder' not in pose_points:
            return None

        ls = pose_points['left_shoulder']
        rs = pose_points['right_shoulder']
        anchor = (ls + rs) / 2.0
        scale  = calculate_distance(ls, rs)

        if scale < 1e-6:
            return None

        return {'anchor': anchor, 'scale': scale}

    def _normalize_landmarks(self, pose_points: Dict,
                             norm_params: Dict,
                             timestamp: float) -> Dict:
        anchor = norm_params['anchor']
        scale  = norm_params['scale']
        normalized = {}

        for name, point in pose_points.items():
            norm_point = (point - anchor) / scale

            if self.smoothing_enabled and f'{name}_x' in self.filters:
                x = self.filters[f'{name}_x'](norm_point[0], timestamp)
                y = self.filters[f'{name}_y'](norm_point[1], timestamp)
                z = self.filters[f'{name}_z'](norm_point[2], timestamp)
                norm_point = np.array([x, y, z])

            normalized[name] = norm_point

        return normalized

    def _calculate_bone_transforms(self, normalized_pose: Dict) -> Dict:
        bones = {}

        # Just send joint positions directly
        # Godot will do the rotation math itself
        joint_map = {
            'left_shoulder':  'pos_left_shoulder',
            'right_shoulder': 'pos_right_shoulder',
            'left_elbow':     'pos_left_elbow',
            'right_elbow':    'pos_right_elbow',
            'left_wrist':     'pos_left_wrist',
            'right_wrist':    'pos_right_wrist',
        }

        for pose_key, bone_key in joint_map.items():
            if pose_key not in normalized_pose:
                continue
            v = normalized_pose[pose_key]
            bones[bone_key] = {
                'x': float(v[0]),
                'y': float(v[1]),
                'z': float(v[2])
            }

        return bones

    def _calculate_finger_transforms(self, hand_landmarks: Dict,
                                     side: str) -> Dict:
        bones = {}

        if len(hand_landmarks) < 5:
            return bones

        finger_map = {
            'thumb':  [1, 2, 3, 4],
            'index':  [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring':   [13, 14, 15, 16],
            'pinky':  [17, 18, 19, 20],
        }

        if side == 'left':
            bone_names = {
                'thumb':  ['mixamorig_LeftHandThumb1',  'mixamorig_LeftHandThumb2',  'mixamorig_LeftHandThumb3'],
                'index':  ['mixamorig_LeftHandIndex1',  'mixamorig_LeftHandIndex2',  'mixamorig_LeftHandIndex3'],
                'middle': ['mixamorig_LeftHandMiddle1', 'mixamorig_LeftHandMiddle2', 'mixamorig_LeftHandMiddle3'],
                'ring':   ['mixamorig_LeftHandRing1',   'mixamorig_LeftHandRing2',   'mixamorig_LeftHandRing3'],
                'pinky':  ['mixamorig_LeftHandPinky1',  'mixamorig_LeftHandPinky2',  'mixamorig_LeftHandPinky3'],
            }
            ref_vec = np.array([-1, 0, 0])
        else:
            bone_names = {
                'thumb':  ['mixamorig_RightHandThumb1',  'mixamorig_RightHandThumb2',  'mixamorig_RightHandThumb3'],
                'index':  ['mixamorig_RightHandIndex1',  'mixamorig_RightHandIndex2',  'mixamorig_RightHandIndex3'],
                'middle': ['mixamorig_RightHandMiddle1', 'mixamorig_RightHandMiddle2', 'mixamorig_RightHandMiddle3'],
                'ring':   ['mixamorig_RightHandRing1',   'mixamorig_RightHandRing2',   'mixamorig_RightHandRing3'],
                'pinky':  ['mixamorig_RightHandPinky1',  'mixamorig_RightHandPinky2',  'mixamorig_RightHandPinky3'],
            }
            ref_vec = np.array([1, 0, 0])

        def get_point(idx):
            key = str(idx)
            if key not in hand_landmarks:
                return None
            lm = hand_landmarks[key]
            return np.array([lm['x'], -lm['y'], -lm['z']])

        for finger_name, indices in finger_map.items():
            for seg in range(3):
                p1 = get_point(indices[seg])
                p2 = get_point(indices[seg + 1])
                if p1 is None or p2 is None:
                    continue
                seg_vec = normalize_vector(p2 - p1)
                seg_rot = vector_to_quaternion(ref_vec, seg_vec)
                bone_key = f'finger_{side}_{finger_name}_{seg}'
                bones[bone_key] = {
                    'rotation':  seg_rot.tolist(),
                    'bone_name': bone_names[finger_name][seg]
                }

        return bones

    def get_stats(self) -> Dict:
        total = self.processed_count + self.failed_count
        rate  = (self.processed_count / total * 100) if total > 0 else 0
        return {
            'processed':    self.processed_count,
            'failed':       self.failed_count,
            'success_rate': rate
        }