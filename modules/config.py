"""
config.py
---------
Central configuration for the Stereo Vision Depth Estimation pipeline.

Holds simulated camera intrinsics/extrinsics (since no physical calibration
rig is available for this coursework project) and all tunable parameters
for feature matching, RANSAC-based fundamental matrix estimation, and
disparity/depth computation.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Simulated camera intrinsics (typical for a 1282x1110 consumer stereo rig).
# In a real deployment these come from a checkerboard calibration routine
# (cv2.calibrateCamera). We simulate plausible values here so the pipeline
# can still demonstrate the full projection-matrix / depth-from-disparity
# theory covered in the course.
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 1282
IMAGE_HEIGHT = 1110

FOCAL_LENGTH_PX = 1100.0          # focal length in pixels
BASELINE_M = 0.20                 # distance between the two camera centers (m)
CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0

# Intrinsic camera matrix K (same for both cameras -> rectified stereo rig)
K = np.array([
    [FOCAL_LENGTH_PX, 0,               CX],
    [0,               FOCAL_LENGTH_PX, CY],
    [0,               0,               1.0]
], dtype=np.float64)

# Distortion coefficients (assumed negligible / pre-corrected)
DIST_COEFFS = np.zeros(5, dtype=np.float64)

# ---------------------------------------------------------------------------
# Feature matching parameters
# ---------------------------------------------------------------------------
MAX_FEATURES = 5000
LOWE_RATIO = 0.75

# ---------------------------------------------------------------------------
# RANSAC parameters (used for Fundamental / Essential matrix estimation)
# ---------------------------------------------------------------------------
RANSAC_REPROJ_THRESHOLD = 1.0      # px, epipolar-line distance threshold
RANSAC_CONFIDENCE = 0.99
RANSAC_MAX_ITERS = 2000

# ---------------------------------------------------------------------------
# Stereo matching (disparity) parameters
# ---------------------------------------------------------------------------
NUM_DISPARITIES = 16 * 8           # must be divisible by 16
BLOCK_SIZE = 7
UNIQUENESS_RATIO = 10
SPECKLE_WINDOW_SIZE = 100
SPECKLE_RANGE = 32
DISP12_MAX_DIFF = 1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEFT_IMAGE_PATH = "data/aloeL.jpg"
RIGHT_IMAGE_PATH = "data/aloeR.jpg"
OUTPUT_DIR = "outputs"
