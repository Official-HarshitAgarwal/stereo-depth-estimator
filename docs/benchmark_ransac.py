"""
benchmark_ransac.py
---------------------
Quantitatively compares the custom RANSAC Fundamental-matrix estimator
(modules/epipolar.py) against OpenCV's built-in cv2.findFundamentalMat
(also RANSAC-based) on the same feature correspondences.

This exists to back up the "Design Decisions & Rationale" claim in the
project report with actual numbers rather than an assertion — the same
input matches are fed to both estimators and scored on:
  - Inlier count / ratio
  - Mean epipolar (Sampson) constraint residual on the inlier set
  - Wall-clock runtime

Run:  python3 docs/benchmark_ransac.py
"""

import time
import json
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import config, epipolar, evaluate
from modules.utils import get_logger, load_image

logger = get_logger("benchmark")


def run_custom(pts1: np.ndarray, pts2: np.ndarray) -> dict:
    t0 = time.perf_counter()
    F, mask = epipolar.estimate_fundamental_ransac(pts1, pts2)
    elapsed = time.perf_counter() - t0

    residual = evaluate.epipolar_constraint_error(F, pts1[mask], pts2[mask])
    return {
        "implementation": "custom (hand-rolled RANSAC)",
        "inliers": int(np.sum(mask)),
        "inlier_ratio_pct": round(100 * np.sum(mask) / len(pts1), 2),
        "mean_epipolar_residual": round(residual["mean_abs_residual"], 4),
        "fundamental_matrix_rank": int(np.linalg.matrix_rank(F)),
        "runtime_sec": round(elapsed, 4),
    }


def run_opencv(pts1: np.ndarray, pts2: np.ndarray) -> dict:
    t0 = time.perf_counter()
    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC,
        ransacReprojThreshold=config.RANSAC_REPROJ_THRESHOLD,
        confidence=config.RANSAC_CONFIDENCE,
    )
    elapsed = time.perf_counter() - t0
    mask = mask.ravel().astype(bool)

    residual = evaluate.epipolar_constraint_error(F, pts1[mask], pts2[mask])
    return {
        "implementation": "cv2.findFundamentalMat (OpenCV built-in)",
        "inliers": int(np.sum(mask)),
        "inlier_ratio_pct": round(100 * np.sum(mask) / len(pts1), 2),
        "mean_epipolar_residual": round(residual["mean_abs_residual"], 4),
        "fundamental_matrix_rank": int(np.linalg.matrix_rank(F)),
        "runtime_sec": round(elapsed, 4),
    }


def main():
    gray_left = load_image(config.LEFT_IMAGE_PATH, color=False)
    gray_right = load_image(config.RIGHT_IMAGE_PATH, color=False)

    _, _, _, pts1, pts2 = epipolar.detect_and_match_features(gray_left, gray_right)
    logger.info("Benchmarking on %d shared feature correspondences", len(pts1))

    # Average over several runs since RANSAC has run-to-run sampling variance
    n_runs = 5
    custom_runs = [run_custom(pts1, pts2) for _ in range(n_runs)]
    opencv_runs = [run_opencv(pts1, pts2) for _ in range(n_runs)]

    def average(runs):
        avg = dict(runs[0])
        for key in ("inliers", "inlier_ratio_pct", "mean_epipolar_residual", "runtime_sec"):
            avg[key] = round(float(np.mean([r[key] for r in runs])), 4)
        return avg

    result = {
        "num_correspondences": len(pts1),
        "num_runs_averaged": n_runs,
        "custom_ransac": average(custom_runs),
        "opencv_ransac": average(opencv_runs),
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
