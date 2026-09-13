"""
calibration.py
---------------
Module 1: Camera Calibration & Projection Model

Responsible for:
  - Holding/deriving intrinsic camera matrix K
  - Constructing the extrinsic transform [R | t] between the two cameras
  - Building the 3x4 projection matrices P_left, P_right (world -> pixel)
  - Undistorting / rectifying input frames so epipolar lines become
    horizontal scanlines (required for block-matching disparity search)

This directly implements the "projection models" and camera-geometry
theory from the Computer Vision syllabus.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from . import config
from .utils import get_logger

logger = get_logger(__name__)


class StereoCalibration:
    """Encapsulates intrinsic/extrinsic parameters and rectification maps."""

    def __init__(self, K: Optional[np.ndarray] = None, dist: Optional[np.ndarray] = None,
                 baseline_m: Optional[float] = None) -> None:
        self.K = K if K is not None else config.K.copy()
        self.dist = dist if dist is not None else config.DIST_COEFFS.copy()
        self.baseline_m = baseline_m if baseline_m is not None else config.BASELINE_M

        # Extrinsics: right camera is translated by -baseline along X
        # relative to the left camera (standard rectified stereo rig).
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.array([[-self.baseline_m], [0.0], [0.0]], dtype=np.float64)

        self.P_left = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P_right = self.K @ np.hstack([self.R, self.t])

        logger.info("Initialized calibration: f=%.1fpx, baseline=%.3fm",
                    self.K[0, 0], self.baseline_m)

    def projection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (P_left, P_right) 3x4 projection matrices."""
        return self.P_left, self.P_right

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """Remove lens distortion (no-op with zero distortion coeffs, but
        kept so the pipeline is correct for real, uncorrected camera feeds)."""
        return cv2.undistort(img, self.K, self.dist)

    def rectify_pair(self, img_left: np.ndarray, img_right: np.ndarray,
                      R_rel: np.ndarray, t_rel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stereo-rectify a pair given a *measured* relative rotation/translation
        (typically the output of recoverPose in epipolar.py). Produces images
        whose corresponding epipolar lines are horizontal, which is a
        precondition for standard block-matching disparity estimation.
        """
        h, w = img_left.shape[:2]
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K, self.dist, self.K, self.dist,
            (w, h), R_rel, t_rel, alpha=0
        )
        map1x, map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(
            self.K, self.dist, R2, P2, (w, h), cv2.CV_32FC1)

        rect_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right, Q
