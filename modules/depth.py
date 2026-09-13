"""
depth.py
--------
Module 3: Disparity & Depth Estimation

Responsible for:
  - Computing a dense disparity map from a rectified stereo pair (SGBM)
  - Converting disparity -> metric depth using the classic
        Z = (f * B) / d
    relation covered in the stereo-vision syllabus section
  - Basic post-processing (speckle filtering is handled by SGBM params)
  - Depth statistics/evaluation helpers
"""

import numpy as np
import cv2

from . import config
from .utils import get_logger

logger = get_logger(__name__)


def compute_disparity(img_left_gray: np.ndarray, img_right_gray: np.ndarray) -> np.ndarray:
    """Compute a dense disparity map using Semi-Global Block Matching (SGBM)."""
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=config.NUM_DISPARITIES,
        blockSize=config.BLOCK_SIZE,
        P1=8 * 3 * config.BLOCK_SIZE ** 2,
        P2=32 * 3 * config.BLOCK_SIZE ** 2,
        disp12MaxDiff=config.DISP12_MAX_DIFF,
        uniquenessRatio=config.UNIQUENESS_RATIO,
        speckleWindowSize=config.SPECKLE_WINDOW_SIZE,
        speckleRange=config.SPECKLE_RANGE,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    raw_disp = stereo.compute(img_left_gray, img_right_gray).astype(np.float32) / 16.0
    raw_disp[raw_disp <= 0.0] = np.nan  # mask invalid / unmatched pixels

    valid_pct = 100 * np.sum(~np.isnan(raw_disp)) / raw_disp.size
    logger.info("Disparity computed: %.1f%% of pixels have valid matches", valid_pct)

    return raw_disp


def disparity_to_depth(disparity: np.ndarray, focal_length_px: float = None,
                        baseline_m: float = None) -> np.ndarray:
    """
    Convert a disparity map to a metric depth map using:
        depth (m) = (focal_length_px * baseline_m) / disparity_px
    """
    f = focal_length_px if focal_length_px is not None else config.FOCAL_LENGTH_PX
    b = baseline_m if baseline_m is not None else config.BASELINE_M

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (f * b) / disparity

    depth[~np.isfinite(depth)] = np.nan
    return depth


def depth_statistics(depth: np.ndarray) -> dict:
    """Basic evaluation metrics used in the report's Evaluation Methodology."""
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return {"valid_pixel_pct": 0.0}
    return {
        "valid_pixel_pct": round(100 * valid.size / depth.size, 2),
        "min_depth_m": round(float(np.min(valid)), 3),
        "max_depth_m": round(float(np.percentile(valid, 99)), 3),
        "mean_depth_m": round(float(np.mean(valid)), 3),
        "median_depth_m": round(float(np.median(valid)), 3),
    }
