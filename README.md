# Stereo Vision Depth Estimator

A from-scratch stereo vision pipeline that recovers a metric depth map from a
pair of left/right camera images, built for the **Computer Vision (CSE3010)**
"Build Your Own Project" evaluation.

The pipeline is not a thin wrapper around one OpenCV call — the Fundamental
matrix is estimated with a **hand-implemented RANSAC loop** (normalized
8-point algorithm + Sampson-distance inlier scoring + adaptive trial count),
so the core algorithm covered in the syllabus is actually implemented, not
just invoked.

## Overview

Given two images of the same scene taken from slightly different horizontal
viewpoints, the system:

1. Detects and matches SIFT keypoints between the views
2. Robustly estimates the epipolar geometry (Fundamental / Essential matrix)
   using RANSAC, rejecting mismatched features as outliers
3. Recovers the relative camera pose and rectifies the image pair so
   corresponding points lie on the same horizontal scanline
4. Computes a dense disparity map via Semi-Global Block Matching (SGBM)
5. Converts disparity to metric depth using `Z = (f * B) / d`
6. Emits visual diagnostics and a JSON summary of quality metrics at every stage

## Features

- **Custom RANSAC Fundamental-matrix estimation** — normalized 8-point
  algorithm, Sampson geometric error, adaptive iteration count (not a single
  call to `cv2.findFundamentalMat`)
- **Full camera-geometry pipeline** — intrinsic/extrinsic matrices, projection
  matrices, Essential matrix derivation, pose recovery, stereo rectification
- **Dense depth estimation** — SGBM disparity + disparity-to-depth conversion
  with metric (meters) output
- **Rich visual diagnostics** — feature matches, RANSAC inlier overlay,
  epipolar line overlay, rectified pair, disparity heatmap, depth heatmap
- **Quantitative evaluation** — epipolar constraint residuals, inlier ratio,
  disparity coverage %, depth validity %, all written to `summary_metrics.json`
- **9 automated tests** (pytest) covering calibration, feature matching,
  RANSAC correctness (rank-2 constraint, inlier ratio), disparity/depth
  conversion, and error handling

## Technologies / Tools Used

| Category | Tool |
|---|---|
| Language | Python 3.12 |
| Computer Vision | OpenCV (`opencv-python`) — SIFT, BFMatcher, SGBM, rectification |
| Numerical | NumPy |
| Visualization | Matplotlib, OpenCV colormaps |
| Testing | pytest |
| Diagramming | Graphviz |
| Version control | Git / GitHub |

## Project Structure

```
stereo-depth-estimator/
├── main.py                    # Pipeline orchestrator (CLI entry point)
├── modules/
│   ├── config.py               # Camera intrinsics/extrinsics & tunables
│   ├── calibration.py          # Module 1: projection model, rectification
│   ├── epipolar.py              # Module 2: SIFT matching + RANSAC epipolar geometry
│   ├── depth.py                 # Module 3: SGBM disparity + depth conversion
│   ├── evaluate.py               # Quantitative validation metrics
│   ├── visualize.py              # Match/epiline overlay rendering
│   └── utils.py                  # I/O, logging, colormap helpers
├── tests/
│   └── test_pipeline.py          # 9 pytest unit/validation tests
├── data/
│   ├── aloeL.jpg                 # Sample stereo pair (Middlebury "Aloe" set)
│   └── aloeR.jpg
├── docs/
│   ├── generate_diagrams.py      # Regenerates all design diagrams
│   └── diagrams/                 # Architecture / workflow / UML PNGs
├── outputs/                       # Pipeline run outputs (generated)
├── statement.md
└── README.md
```

## Steps to Install & Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd stereo-depth-estimator

# 2. Install dependencies
pip install opencv-python numpy matplotlib pytest graphviz

# 3. Run the pipeline on the included sample stereo pair
python3 main.py --left data/aloeL.jpg --right data/aloeR.jpg --outdir outputs

# 4. (Optional) Run on your own stereo pair
python3 main.py --left path/to/left.jpg --right path/to/right.jpg --outdir outputs
```

Outputs are written to `outputs/`:
| File | Description |
|---|---|
| `01_input_side_by_side.png` | Raw input pair |
| `02_feature_matches.png` | SIFT matches before RANSAC filtering |
| `03_ransac_inlier_matches.png` | Matches surviving RANSAC (green) |
| `04_epipolar_lines.png` | Epipolar line overlay on the right image |
| `05_rectified_pair.png` | Stereo-rectified pair (horizontal epipolar lines) |
| `06_disparity_map.png` | Pseudo-colored disparity heatmap |
| `07_depth_map.png` | Pseudo-colored metric depth heatmap |
| `summary_metrics.json` | All quantitative evaluation metrics |

## Instructions for Testing

```bash
pip install pytest
pytest tests/ -v
```

9 tests validate: calibration matrix shapes, feature-matching correctness,
RANSAC's rank-2 constraint on the Fundamental matrix, RANSAC inlier ratio
sanity, the epipolar constraint residual, disparity coverage, the
disparity→depth inverse relationship, correct NaN-handling at zero
disparity, and clear error handling for missing input files.

## Regenerating the Design Diagrams

```bash
python3 docs/generate_diagrams.py
```

## Screenshots

See `outputs/06_disparity_map.png` and `outputs/07_depth_map.png` for example
results on the included Middlebury "Aloe" stereo pair — the potted plant
(near) is clearly separated from the patterned background (far) in the
depth heatmap.

## Non-Functional Requirements Addressed

- **Performance** — full pipeline (feature matching → depth map) completes in
  under 4 seconds on a 1282×1110 stereo pair on a standard CPU
- **Reliability** — RANSAC uses an adaptive-but-floored trial-count schedule
  so a single unlucky minimal sample cannot silently degrade the fit
- **Error handling** — explicit `FileNotFoundError` / `ValueError` on bad
  input paths or undecodable images, with clear CLI-facing messages
- **Logging/monitoring** — every pipeline stage logs progress, match counts,
  inlier ratios, and disparity coverage via Python's `logging` module
- **Maintainability** — one responsibility per module (calibration / epipolar
  geometry / depth), each independently unit-tested
- **Resource efficiency** — disparity computed once per run at native
  resolution; no redundant recomputation across stages

## Future Enhancements

- Replace simulated intrinsics with a real OpenCV checkerboard calibration routine
- Add a Streamlit UI for interactive threshold/parameter tuning
- Extend to a full Structure-from-Motion pipeline (multi-view triangulation)
- GPU-accelerated SGBM for real-time video depth estimation

## References

1. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, Cambridge University Press.
2. OpenCV documentation — Camera Calibration and 3D Reconstruction.
3. D. Scharstein and R. Szeliski, Middlebury Stereo Datasets.
4. M. A. Fischler and R. C. Bolles, "Random Sample Consensus," *Communications of the ACM*, 1981.
