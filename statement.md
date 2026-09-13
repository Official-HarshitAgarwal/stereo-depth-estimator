# Problem Statement

## Problem Statement

Estimating the distance of objects from a camera system — depth — is a core
problem in computer vision, robotics, and autonomous navigation. A single
camera image contains no direct depth information; depth must be inferred
geometrically. Given a stereo camera rig (two cameras with a known horizontal
offset), corresponding points in the left and right images are shifted
horizontally by an amount ("disparity") that is inversely proportional to the
scene depth at that point. Recovering this relationship reliably from
real-world images — where feature matches contain outliers, lens distortion
exists, and images are not naturally aligned — requires correctly applying
projection models, epipolar geometry, and robust estimation (RANSAC).

This project builds a complete stereo vision pipeline that takes a raw,
unrectified stereo image pair and produces a dense, per-pixel metric depth
map, while making every intermediate geometric step (feature correspondence,
Fundamental/Essential matrix estimation, pose recovery, rectification)
inspectable and testable.

## Scope of the Project

**In scope:**
- Simulated (non-measured) camera intrinsic parameters, since no physical
  calibration rig is available for this coursework project
- Two-view (binocular) stereo depth estimation from static image pairs
- Robust epipolar geometry estimation via a custom RANSAC implementation
- Dense disparity computation via Semi-Global Block Matching
- Quantitative validation of each pipeline stage

**Out of scope:**
- Real-time video / multi-frame temporal depth estimation
- Multi-view (3+ camera) Structure-from-Motion or full 3D reconstruction
- Physical camera calibration using a checkerboard rig
- Deep-learning-based monocular depth estimation

## Target Users

- Computer vision students / evaluators reviewing the correctness of
  epipolar-geometry and RANSAC implementations
- Robotics/embedded-vision developers prototyping a stereo depth module
  before deploying against a physically calibrated rig

## High-Level Features

1. **Camera calibration & projection modeling** — intrinsic matrix, extrinsic
   transform, 3×4 projection matrices for both views
2. **Robust epipolar geometry estimation** — SIFT feature matching + a custom
   RANSAC loop (normalized 8-point algorithm, Sampson-distance scoring,
   adaptive iteration count) to estimate the Fundamental matrix, followed by
   Essential-matrix derivation and relative pose recovery
3. **Dense depth estimation** — stereo rectification, SGBM disparity
   computation, and disparity-to-metric-depth conversion
4. **Evaluation & diagnostics** — epipolar constraint residuals, inlier
   ratios, disparity/depth coverage statistics, all exported as JSON
