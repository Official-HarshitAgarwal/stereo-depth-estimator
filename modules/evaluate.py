"""
evaluate.py
-----------
Evaluation utilities used to validate each pipeline stage and to produce
the numbers quoted in the project report's "Testing Approach" section.
"""

import numpy as np


def epipolar_constraint_error(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> dict:
    """
    Sanity-check the estimated Fundamental matrix: for a perfect F,
    x2^T F x1 == 0 for every true correspondence. We report the residual
    distribution as a validation metric.
    """
    p1h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    p2h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    residuals = np.abs(np.sum((p2h @ F) * p1h, axis=1))
    return {
        "mean_abs_residual": float(np.mean(residuals)),
        "median_abs_residual": float(np.median(residuals)),
        "max_abs_residual": float(np.max(residuals)),
    }


def inlier_ratio(mask: np.ndarray) -> float:
    mask = np.asarray(mask).ravel()
    return float(np.sum(mask > 0) / len(mask)) if len(mask) else 0.0


def disparity_coverage(disparity: np.ndarray) -> float:
    """Fraction of pixels for which SGBM produced a valid disparity value."""
    return float(100 * np.sum(np.isfinite(disparity)) / disparity.size)


def run_unit_style_checks(F, mask_inliers, disparity, depth) -> dict:
    """Aggregate a small report-friendly dict of validation results."""
    return {
        "ransac_inlier_ratio_pct": round(100 * inlier_ratio(mask_inliers), 2),
        "disparity_valid_coverage_pct": round(disparity_coverage(disparity), 2),
        "fundamental_matrix_rank": int(np.linalg.matrix_rank(F)),
        "depth_valid_pixels_pct": round(
            100 * np.sum(np.isfinite(depth)) / depth.size, 2
        ),
    }
