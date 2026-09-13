"""
test_pipeline.py
-----------------
Unit / validation tests for the stereo depth estimation pipeline.
Run with:  pytest tests/ -v
"""

import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import config, calibration, epipolar, depth as depth_mod, evaluate


LEFT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aloeL.jpg")
RIGHT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aloeR.jpg")


@pytest.fixture(scope="module")
def stereo_images():
    left = cv2.imread(LEFT_PATH, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(RIGHT_PATH, cv2.IMREAD_GRAYSCALE)
    assert left is not None and right is not None, "Test data images missing"
    return left, right


def test_calibration_matrix_shapes():
    calib = calibration.StereoCalibration()
    P_left, P_right = calib.projection_matrices()
    assert P_left.shape == (3, 4)
    assert P_right.shape == (3, 4)
    assert calib.K.shape == (3, 3)


def test_feature_matching_produces_matches(stereo_images):
    left, right = stereo_images
    kp1, kp2, matches, pts1, pts2 = epipolar.detect_and_match_features(left, right)
    assert len(kp1) > 0 and len(kp2) > 0
    assert len(matches) >= 8, "Need at least 8 matches for fundamental matrix estimation"
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2


def test_ransac_fundamental_matrix_rank2(stereo_images):
    left, right = stereo_images
    _, _, _, pts1, pts2 = epipolar.detect_and_match_features(left, right)
    F, inlier_mask = epipolar.estimate_fundamental_ransac(pts1, pts2)

    assert F.shape == (3, 3)
    # Fundamental matrix must be rank-2 by construction
    rank = np.linalg.matrix_rank(F, tol=1e-3)
    assert rank == 2
    assert np.sum(inlier_mask) >= 8
    assert np.sum(inlier_mask) <= len(pts1)


def test_ransac_finds_reasonable_inlier_ratio(stereo_images):
    left, right = stereo_images
    _, _, _, pts1, pts2 = epipolar.detect_and_match_features(left, right)
    F, inlier_mask = epipolar.estimate_fundamental_ransac(pts1, pts2)
    ratio = evaluate.inlier_ratio(inlier_mask)
    # Aloe pair is a clean, well-textured Middlebury set: expect a solid majority
    assert ratio > 0.4, f"Unexpectedly low inlier ratio: {ratio:.2f}"


def test_epipolar_constraint_residual_is_small(stereo_images):
    left, right = stereo_images
    _, _, _, pts1, pts2 = epipolar.detect_and_match_features(left, right)
    F, inlier_mask = epipolar.estimate_fundamental_ransac(pts1, pts2)
    result = evaluate.epipolar_constraint_error(F, pts1[inlier_mask], pts2[inlier_mask])
    assert result["mean_abs_residual"] < 1.0


def test_disparity_map_has_valid_coverage(stereo_images):
    left, right = stereo_images
    disparity = depth_mod.compute_disparity(left, right)
    coverage = evaluate.disparity_coverage(disparity)
    assert coverage > 20.0, f"Too few valid disparity pixels: {coverage:.1f}%"


def test_disparity_to_depth_conversion_monotonic():
    """Smaller disparity must yield larger depth (inverse relationship)."""
    disp_near = np.array([[40.0]])
    disp_far = np.array([[10.0]])
    depth_near = depth_mod.disparity_to_depth(disp_near)
    depth_far = depth_mod.disparity_to_depth(disp_far)
    assert depth_far[0, 0] > depth_near[0, 0]


def test_disparity_to_depth_handles_zero_disparity():
    disp = np.array([[0.0, 5.0]])
    depth = depth_mod.disparity_to_depth(disp)
    assert np.isnan(depth[0, 0])  # zero disparity -> undefined / infinite depth
    assert np.isfinite(depth[0, 1])


def test_missing_image_raises_clear_error():
    from modules.utils import load_image
    with pytest.raises(FileNotFoundError):
        load_image("data/does_not_exist.jpg")
