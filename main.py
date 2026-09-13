"""
main.py
-------
Stereo Vision Depth Estimator — end-to-end pipeline entry point.

Workflow (matches the workflow diagram in docs/):
  1. Load stereo image pair -> undistort
  2. Detect + match SIFT features between views
  3. Robustly estimate the Fundamental matrix using a custom RANSAC loop
  4. Derive the Essential matrix + recover relative camera pose
  5. Stereo-rectify the pair using the recovered pose
  6. Compute dense disparity map (SGBM) on the rectified pair
  7. Convert disparity -> metric depth using camera geometry
  8. Save all intermediate + final visual outputs and print evaluation metrics

Usage:
    python main.py --left data/aloeL.jpg --right data/aloeR.jpg
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np

from modules import config, calibration, epipolar, depth as depth_mod, evaluate, visualize
from modules.utils import get_logger, load_image, ensure_dir, save_image, save_colormap

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo Vision Depth Estimator")
    parser.add_argument("--left", default=config.LEFT_IMAGE_PATH, help="Path to left image")
    parser.add_argument("--right", default=config.RIGHT_IMAGE_PATH, help="Path to right image")
    parser.add_argument("--outdir", default=config.OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


def run_pipeline(left_path: str, right_path: str, outdir: str) -> dict:
    t0 = time.time()
    ensure_dir(outdir)

    # --- Stage 1: Load + undistort -----------------------------------------
    try:
        img_left = load_image(left_path)
        img_right = load_image(right_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Input error: %s", e)
        sys.exit(1)

    calib = calibration.StereoCalibration()
    img_left = calib.undistort(img_left)
    img_right = calib.undistort(img_right)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    save_image(f"{outdir}/01_input_side_by_side.png",
               np.hstack([img_left, img_right]))

    # --- Stage 2: Feature detection + matching ------------------------------
    kp1, kp2, matches, pts1, pts2 = epipolar.detect_and_match_features(gray_left, gray_right)
    if len(matches) < 8:
        logger.error("Not enough matches (%d) to estimate geometry.", len(matches))
        sys.exit(1)

    match_vis = visualize.draw_matches(img_left, kp1, img_right, kp2, matches)
    save_image(f"{outdir}/02_feature_matches.png", match_vis)

    # --- Stage 3: RANSAC Fundamental matrix ---------------------------------
    F, inlier_mask = epipolar.estimate_fundamental_ransac(pts1, pts2)

    inlier_vis = visualize.draw_matches(img_left, kp1, img_right, kp2, matches, inlier_mask)
    save_image(f"{outdir}/03_ransac_inlier_matches.png", inlier_vis)

    epi_check = evaluate.epipolar_constraint_error(F, pts1[inlier_mask], pts2[inlier_mask])

    lines_right = epipolar.compute_epipolar_lines(F, pts1[inlier_mask], which_image=1)
    epi_vis = visualize.draw_epilines(img_right, lines_right, pts2[inlier_mask])
    save_image(f"{outdir}/04_epipolar_lines.png", epi_vis)

    # --- Stage 4: Essential matrix + relative pose --------------------------
    E, R, t, pose_mask = epipolar.recover_relative_pose(
        F, pts1[inlier_mask], pts2[inlier_mask], calib.K)

    # --- Stage 5: Stereo rectification --------------------------------------
    rect_left, rect_right, Q = calib.rectify_pair(img_left, img_right, R, t)
    save_image(f"{outdir}/05_rectified_pair.png", np.hstack([rect_left, rect_right]))

    rect_gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    rect_gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    # --- Stage 6: Disparity ---------------------------------------------------
    disparity = depth_mod.compute_disparity(rect_gray_left, rect_gray_right)
    save_colormap(f"{outdir}/06_disparity_map.png", disparity)

    # --- Stage 7: Depth --------------------------------------------------------
    depth_map = depth_mod.disparity_to_depth(disparity)
    save_colormap(f"{outdir}/07_depth_map.png", depth_map, colormap=cv2.COLORMAP_TURBO)

    depth_stats = depth_mod.depth_statistics(depth_map)

    # --- Stage 8: Evaluation summary --------------------------------------------
    checks = evaluate.run_unit_style_checks(F, pose_mask, disparity, depth_map)

    summary = {
        "runtime_sec": round(time.time() - t0, 2),
        "num_matches": len(matches),
        "num_ransac_inliers": int(np.sum(inlier_mask)),
        "epipolar_constraint_check": epi_check,
        "depth_statistics": depth_stats,
        "quality_checks": checks,
    }

    with open(f"{outdir}/summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Pipeline complete in %.2fs. Metrics written to %s/summary_metrics.json",
                summary["runtime_sec"], outdir)
    return summary


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(args.left, args.right, args.outdir)
    print(json.dumps(result, indent=2))
