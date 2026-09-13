"""
visualize.py
------------
Rendering helpers for human-inspectable outputs: keypoint match overlays,
epipolar-line overlays, and side-by-side rectified pairs. Kept separate from
the numerical modules so the core algorithms remain unit-testable without a
display/plotting dependency.
"""

import cv2
import numpy as np


def draw_matches(img_left, kp1, img_right, kp2, matches, inlier_mask=None, max_draw=60):
    """Draw a subsample of matches, colored green for RANSAC inliers."""
    if inlier_mask is not None:
        matches = [m for m, keep in zip(matches, inlier_mask) if keep]
    matches = matches[:max_draw]
    return cv2.drawMatches(
        img_left, kp1, img_right, kp2, matches, None,
        matchColor=(60, 200, 60), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_epilines(img: np.ndarray, lines: np.ndarray, pts: np.ndarray, max_lines=25) -> np.ndarray:
    """Overlay a subset of epipolar lines + their corresponding points on img."""
    out = img.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(7)

    idxs = rng.choice(len(lines), size=min(max_lines, len(lines)), replace=False)
    for i in idxs:
        a, b, c = lines[i]
        color = tuple(int(x) for x in rng.integers(0, 255, 3))
        x0, y0 = 0, int(-c / b) if b != 0 else 0
        x1, y1 = w, int(-(c + a * w) / b) if b != 0 else h
        cv2.line(out, (x0, y0), (x1, y1), color, 1)
        cv2.circle(out, tuple(pts[i].astype(int)), 5, color, -1)
    return out
