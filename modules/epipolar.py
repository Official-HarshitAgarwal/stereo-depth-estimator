"""
epipolar.py
------------
Module 2: Feature Matching & Epipolar Geometry (RANSAC-based)

Responsible for:
  - Detecting and matching keypoints between the left/right views (SIFT + BF/Lowe ratio)
  - Estimating the Fundamental matrix F robustly via a hand-implemented
    RANSAC loop (not just calling cv2.findFundamentalMat blindly) to
    demonstrate understanding of the RANSAC algorithm from the syllabus
  - Deriving the Essential matrix E = K^T F K and recovering relative pose (R, t)
  - Computing epipolar lines for visualization/validation

This is the algorithmic core of the "epipolar geometry" and "RANSAC"
topics from the Computer Vision course.
"""

import random
from typing import Optional, Tuple

import numpy as np
import cv2

from . import config
from .utils import get_logger

logger = get_logger(__name__)


def detect_and_match_features(img_left_gray: np.ndarray, img_right_gray: np.ndarray) -> Tuple[
        list, list, list, np.ndarray, np.ndarray]:
    """SIFT feature detection + Lowe's-ratio-filtered brute-force matching."""
    sift = cv2.SIFT_create(nfeatures=config.MAX_FEATURES)
    kp1, des1 = sift.detectAndCompute(img_left_gray, None)
    kp2, des2 = sift.detectAndCompute(img_right_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError("Feature detection failed on one or both images.")

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < config.LOWE_RATIO * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    logger.info("Feature matching: %d keypoints (L) / %d (R), %d good matches "
                "after ratio test", len(kp1), len(kp2), len(good_matches))

    return kp1, kp2, good_matches, pts1, pts2


def _fundamental_from_8_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Normalized 8-point algorithm for a minimal Fundamental matrix estimate."""
    def normalize(pts):
        centroid = pts.mean(axis=0)
        shifted = pts - centroid
        mean_dist = np.mean(np.linalg.norm(shifted, axis=1)) + 1e-9
        scale = np.sqrt(2) / mean_dist
        T = np.array([[scale, 0, -scale * centroid[0]],
                      [0, scale, -scale * centroid[1]],
                      [0, 0, 1]])
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        norm_pts = (T @ pts_h.T).T
        return norm_pts, T

    n1, T1 = normalize(p1)
    n2, T2 = normalize(p2)

    A = np.zeros((n1.shape[0], 9))
    for i in range(n1.shape[0]):
        x1, y1 = n1[i, 0], n1[i, 1]
        x2, y2 = n2[i, 0], n2[i, 1]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt2 = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt2

    F = T2.T @ F @ T1
    return F / (F[2, 2] + 1e-12)


def _sampson_distance(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Geometric (Sampson) error used to score RANSAC inliers/outliers."""
    p1h = np.hstack([p1, np.ones((p1.shape[0], 1))])
    p2h = np.hstack([p2, np.ones((p2.shape[0], 1))])

    Fx1 = (F @ p1h.T).T
    Ftx2 = (F.T @ p2h.T).T
    x2tFx1 = np.sum(p2h * Fx1, axis=1)

    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
    return (x2tFx1 ** 2) / denom


def estimate_fundamental_ransac(pts1: np.ndarray, pts2: np.ndarray,
                                 threshold: float = config.RANSAC_REPROJ_THRESHOLD,
                                 confidence: float = config.RANSAC_CONFIDENCE,
                                 max_iters: int = config.RANSAC_MAX_ITERS,
                                 seed: int = 42):
    """
    Custom RANSAC loop to robustly estimate the Fundamental matrix in the
    presence of outlier matches. Returns the best F and a boolean inlier mask.
    """
    rng = random.Random(seed)
    n_points = pts1.shape[0]
    if n_points < 8:
        raise ValueError("Need at least 8 point correspondences for RANSAC.")

    best_F, best_inliers, best_inlier_count = None, None, 0
    it = 0
    trials_needed = max_iters

    while it < min(max_iters, trials_needed):
        sample_idx = rng.sample(range(n_points), 8)
        try:
            F_candidate = _fundamental_from_8_points(pts1[sample_idx], pts2[sample_idx])
        except np.linalg.LinAlgError:
            it += 1
            continue

        errors = _sampson_distance(F_candidate, pts1, pts2)
        inliers = errors < threshold
        inlier_count = int(np.sum(inliers))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_F = F_candidate
            best_inliers = inliers

            # Adaptively shrink number of trials needed (standard RANSAC formula:
            # N = log(1-p) / log(1 - w^s)). Floor w so a single lucky/unlucky
            # sample early on can't collapse trials_needed to an unsafe value.
            w = max(inlier_count / n_points, 0.05)
            denom = np.log(1 - w ** 8)
            if not np.isfinite(denom) or denom >= 0:
                trials_needed = max_iters
            else:
                trials_needed = int(np.log(1 - confidence) / denom)
            trials_needed = int(np.clip(trials_needed, 1, max_iters))

        it += 1

    if best_F is None:
        raise RuntimeError("RANSAC failed to find a valid Fundamental matrix.")

    # Refit F using all inliers for a more accurate final estimate
    best_F = _fundamental_from_8_points(pts1[best_inliers], pts2[best_inliers])

    logger.info("RANSAC F-matrix: %d/%d inliers (%.1f%%) after %d iterations",
                best_inlier_count, n_points, 100 * best_inlier_count / n_points, it)

    return best_F, best_inliers


def recover_relative_pose(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
                           K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive Essential matrix E from F, then recover relative rotation/translation."""
    E = K.T @ F @ K
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    logger.info("Recovered relative pose from Essential matrix (E = K^T F K)")
    return E, R, t, mask


def compute_epipolar_lines(F: np.ndarray, pts: np.ndarray, which_image: int) -> np.ndarray:
    """Compute epipolar lines in the OTHER image for points in `which_image`."""
    return cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), which_image, F).reshape(-1, 3)
