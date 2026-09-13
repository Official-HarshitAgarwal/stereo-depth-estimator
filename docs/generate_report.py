"""
generate_report.py
--------------------
Builds the full project report PDF (docs/Project_Report.pdf) required by
section 6 of the VITyarthi "Build Your Own Project" specification.
"""

import json
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle,
    ListFlowable, ListItem
)
from reportlab.lib import colors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIAG = os.path.join(ROOT, "docs", "diagrams")
OUT = os.path.join(ROOT, "outputs")
REPORT_PATH = os.path.join(ROOT, "docs", "Project_Report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", fontSize=24, leading=30, alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle(name="CoverSub", fontSize=13, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#444441")))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10.5, leading=15))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#5F5E5A"), alignment=TA_CENTER, spaceAfter=10))

with open(os.path.join(OUT, "summary_metrics.json")) as f:
    metrics = json.load(f)

story = []

# ---------------------------------------------------------------------------
# Cover Page
# ---------------------------------------------------------------------------
story.append(Spacer(1, 5 * cm))
story.append(Paragraph("Stereo Vision Depth Estimator", styles["CoverTitle"]))
story.append(Paragraph("A RANSAC-based Epipolar Geometry Pipeline for Dense Depth Estimation", styles["CoverSub"]))
story.append(Spacer(1, 2 * cm))
story.append(Paragraph("Course: Computer Vision (CSE3010)", styles["CoverSub"]))
story.append(Paragraph("Submitted as part of: VITyarthi - Build Your Own Project", styles["CoverSub"]))
story.append(Spacer(1, 1 * cm))
story.append(Paragraph("Harshit Agarwal (KAIZER)", styles["CoverSub"]))
story.append(Paragraph("B.Tech CSE (AI/ML), VIT Bhopal University", styles["CoverSub"]))
story.append(PageBreak())


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(i, styles["Body"]), leftIndent=12) for i in items],
        bulletType="bullet", start="circle"
    )


def add_image(path, width=15 * cm, caption=None):
    if os.path.exists(path):
        img = Image(path, width=width, height=width * 0.62)
        story.append(img)
        if caption:
            story.append(Paragraph(caption, styles["Caption"]))
    else:
        story.append(Paragraph(f"[Missing image: {path}]", styles["Body"]))


# 1. Introduction
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "Depth perception from images is a fundamental problem in computer vision with "
    "applications in robotics, autonomous navigation, and 3D reconstruction. This project "
    "implements a complete two-view (binocular) stereo vision pipeline that recovers a dense, "
    "per-pixel metric depth map from a stereo image pair, grounded directly in the projection "
    "models, epipolar geometry, and RANSAC concepts covered in the Computer Vision (CSE3010) "
    "syllabus.", styles["Body"]))
story.append(Spacer(1, 6))

# 2. Problem Statement
story.append(Paragraph("2. Problem Statement", styles["H1"]))
story.append(Paragraph(
    "A single 2D image discards depth information inherent to a 3D scene. Stereo vision "
    "recovers this depth by observing the same scene from two horizontally offset viewpoints: "
    "a 3D point projects to horizontally shifted pixel locations in the two images, and this "
    "shift (disparity) is inversely proportional to the point's distance from the camera. "
    "Recovering this relationship reliably requires (a) finding accurate correspondences between "
    "the two views, (b) rejecting incorrect matches robustly, and (c) using the recovered "
    "geometry to rectify and triangulate depth. This project addresses all three sub-problems "
    "end-to-end.", styles["Body"]))

# 3. Functional Requirements
story.append(Paragraph("3. Functional Requirements", styles["H1"]))
story.append(bullets([
    "<b>Module 1 - Calibration & Projection Model:</b> maintain intrinsic camera matrix K, "
    "extrinsic transform [R|t], and 3x4 projection matrices for both views; undistort input frames.",
    "<b>Module 2 - Feature Matching & Epipolar Geometry:</b> detect and match SIFT keypoints; "
    "robustly estimate the Fundamental matrix via a custom RANSAC implementation; derive the "
    "Essential matrix and recover relative camera pose.",
    "<b>Module 3 - Disparity & Depth Estimation:</b> stereo-rectify the pair; compute a dense "
    "disparity map via Semi-Global Block Matching; convert disparity to metric depth.",
    "Clear input/output structure: two image file paths in, a metric depth map plus diagnostic "
    "visualizations and a JSON metrics file out.",
    "A logical, linear workflow: load -> match -> estimate geometry -> rectify -> disparity -> depth -> evaluate.",
]))

# 4. Non-functional Requirements
story.append(Paragraph("4. Non-Functional Requirements", styles["H1"]))
story.append(bullets([
    "<b>Performance:</b> full pipeline completes in under 4 seconds on a 1282x1110 stereo pair on CPU.",
    "<b>Reliability:</b> adaptive-but-floored RANSAC trial scheduling prevents a single unlucky "
    "minimal sample from silently degrading the geometry estimate.",
    "<b>Error handling:</b> explicit, typed exceptions (FileNotFoundError, ValueError, RuntimeError) "
    "with clear messages for missing files, undecodable images, and insufficient matches.",
    "<b>Logging/monitoring:</b> every stage logs progress, match counts, inlier ratios, and disparity "
    "coverage via Python's logging module.",
    "<b>Maintainability:</b> one responsibility per module, each independently unit-tested.",
    "<b>Resource efficiency:</b> disparity computed once per run at native resolution; no redundant recomputation.",
]))

# 5. System Architecture
story.append(Paragraph("5. System Architecture", styles["H1"]))
story.append(Paragraph(
    "The system is organized as three sequential modules feeding an evaluation stage, all "
    "orchestrated by a single pipeline entry point (main.py). Figure 1 shows the high-level "
    "architecture.", styles["Body"]))
add_image(os.path.join(DIAG, "01_system_architecture.png"), caption="Figure 1: System Architecture")
story.append(PageBreak())

# 6. Design Diagrams
story.append(Paragraph("6. Design Diagrams", styles["H1"]))

story.append(Paragraph("6.1 Process Flow / Workflow Diagram", styles["H2"]))
add_image(os.path.join(DIAG, "02_workflow.png"), width=10 * cm, caption="Figure 2: Pipeline Workflow")

story.append(Paragraph("6.2 Use Case Diagram", styles["H2"]))
add_image(os.path.join(DIAG, "03_use_case.png"), caption="Figure 3: Use Case Diagram")

story.append(Paragraph("6.3 Class / Component Diagram", styles["H2"]))
add_image(os.path.join(DIAG, "04_class_diagram.png"), caption="Figure 4: Class/Component Diagram")

story.append(Paragraph("6.4 Sequence Diagram", styles["H2"]))
add_image(os.path.join(DIAG, "05_sequence_diagram.png"), caption="Figure 5: Sequence Diagram of main.run_pipeline()")
story.append(PageBreak())

# 7. Design Decisions & Rationale
story.append(Paragraph("7. Design Decisions & Rationale", styles["H1"]))
story.append(bullets([
    "<b>Custom RANSAC over cv2.findFundamentalMat:</b> implementing the normalized 8-point "
    "algorithm and Sampson-distance inlier scoring by hand demonstrates the algorithmic "
    "understanding required by the syllabus, rather than treating RANSAC as a black box.",
    "<b>SIFT over ORB:</b> SIFT gives more stable, higher-quality correspondences on the "
    "textured Middlebury test scene, which matters more for geometry accuracy than raw speed here.",
    "<b>SGBM over simple block matching:</b> Semi-Global Block Matching produces materially "
    "smoother, more complete disparity maps at a modest computational cost.",
    "<b>Modular package layout:</b> calibration, epipolar geometry, and depth estimation are "
    "separated into independent modules so each can be unit-tested and reasoned about in isolation.",
    "<b>Simulated intrinsics:</b> since no physical stereo rig/checkerboard was available for this "
    "coursework project, plausible fixed intrinsics were used; the code path supports swapping in "
    "real calibration output (via cv2.calibrateCamera) without any interface changes.",
]))

# 8. Implementation Details
story.append(Paragraph("8. Implementation Details", styles["H1"]))
story.append(Paragraph(
    "The pipeline is implemented in Python 3.12 using OpenCV for image processing and "
    "geometric primitives, and NumPy for the custom RANSAC / 8-point algorithm math. The "
    "Fundamental matrix is estimated by a from-scratch loop: (1) sample 8 random correspondences, "
    "(2) solve the normalized 8-point linear system via SVD, (3) enforce the rank-2 constraint "
    "by zeroing the smallest singular value, (4) score all correspondences by Sampson distance, "
    "(5) track the best inlier set, and (6) adaptively shrink the number of remaining trials using "
    "the standard RANSAC formula N = log(1-p) / log(1-w^s). The Essential matrix is derived as "
    "E = K^T F K, and cv2.recoverPose extracts the relative rotation and translation. These are fed "
    "to cv2.stereoRectify to produce a rectified pair on which cv2.StereoSGBM_create computes "
    "disparity, which is converted to metric depth via Z = (f * B) / d.", styles["Body"]))

story.append(Paragraph("8.1 Repository Structure", styles["H2"]))
story.append(Paragraph(
    "modules/config.py, modules/calibration.py, modules/epipolar.py, modules/depth.py, "
    "modules/evaluate.py, modules/visualize.py, modules/utils.py, main.py, tests/test_pipeline.py "
    "(8 source modules + 1 test module = 9 meaningful files, exceeding the minimum requirement).",
    styles["Body"]))

# 9. Dataset Description (ML/computation-heavy section)
story.append(Paragraph("9. Dataset Description", styles["H1"]))
story.append(Paragraph(
    "The pipeline was validated on the Middlebury 'Aloe' stereo pair (1282x1110, distributed "
    "with OpenCV's official sample data), a standard benchmark image pair for stereo matching "
    "research featuring a textured potted plant against a patterned background at multiple depths "
    "- ideal for visually validating disparity/depth separation. The pipeline accepts any "
    "horizontally-offset stereo pair as input via the --left/--right CLI arguments.", styles["Body"]))

story.append(Paragraph("9.1 Model / Algorithm Selection Rationale", styles["H2"]))
story.append(Paragraph(
    "RANSAC was chosen over least-squares fitting for the Fundamental matrix because feature "
    "matching inevitably produces outlier correspondences (repetitive texture, occlusion "
    "boundaries); a single incorrect match can arbitrarily corrupt a least-squares fit, while "
    "RANSAC's inlier-consensus scoring is robust to a majority-inlier match set. SGBM was chosen "
    "over simple block matching for its explicit smoothness penalty terms (P1, P2), which "
    "produce visibly cleaner disparity maps on textured scenes like the Aloe pair.", styles["Body"]))

# 10. Screenshots / Results
story.append(PageBreak())
story.append(Paragraph("10. Screenshots / Results", styles["H1"]))
add_image(os.path.join(OUT, "02_feature_matches.png"), caption="Figure 6: SIFT feature matches before RANSAC filtering")
add_image(os.path.join(OUT, "03_ransac_inlier_matches.png"), caption="Figure 7: RANSAC inlier matches (outliers rejected)")
story.append(PageBreak())
add_image(os.path.join(OUT, "04_epipolar_lines.png"), caption="Figure 8: Epipolar line overlay on the right image")
add_image(os.path.join(OUT, "05_rectified_pair.png"), caption="Figure 9: Stereo-rectified pair")
story.append(PageBreak())
add_image(os.path.join(OUT, "06_disparity_map.png"), width=12 * cm, caption="Figure 10: Disparity map (SGBM)")
add_image(os.path.join(OUT, "07_depth_map.png"), width=12 * cm, caption="Figure 11: Metric depth map")

# Metrics table
story.append(Paragraph("10.1 Quantitative Results", styles["H2"]))
data = [
    ["Metric", "Value"],
    ["Total feature matches (post ratio-test)", str(metrics["num_matches"])],
    ["RANSAC inliers", str(metrics["num_ransac_inliers"])],
    ["RANSAC inlier ratio", f"{metrics['quality_checks']['ransac_inlier_ratio_pct']}%"],
    ["Fundamental matrix rank", str(metrics["quality_checks"]["fundamental_matrix_rank"])],
    ["Mean epipolar constraint residual", f"{metrics['epipolar_constraint_check']['mean_abs_residual']:.4f}"],
    ["Disparity valid coverage", f"{metrics['quality_checks']['disparity_valid_coverage_pct']}%"],
    ["Depth valid pixels", f"{metrics['quality_checks']['depth_valid_pixels_pct']}%"],
    ["Mean estimated depth", f"{metrics['depth_statistics']['mean_depth_m']} m"],
    ["Pipeline runtime", f"{metrics['runtime_sec']} s"],
]
table = Table(data, colWidths=[9 * cm, 6 * cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6F1FB")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B4B2A9")),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1EFE8")]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(table)

# 11. Testing Approach
story.append(Paragraph("11. Testing Approach", styles["H1"]))
story.append(Paragraph(
    "9 automated pytest tests validate correctness at every pipeline stage: calibration matrix "
    "shapes, feature-matching output validity, the Fundamental matrix's mathematically-required "
    "rank-2 property, a sanity floor on RANSAC inlier ratio for a well-textured scene, the "
    "epipolar constraint residual (x2^T F x1 approx 0), disparity map coverage, the inverse "
    "disparity-depth relationship, correct NaN handling at zero disparity, and explicit error "
    "handling for missing input files. All 9 tests pass (see tests/test_pipeline.py).", styles["Body"]))

# 12. Challenges Faced
story.append(Paragraph("12. Challenges Faced", styles["H1"]))
story.append(bullets([
    "The initial adaptive RANSAC trial-count formula could collapse to a single iteration when "
    "an early minimal sample produced a very low (but non-zero) inlier count, due to floating-point "
    "underflow in w^8. Fixed by flooring the inlier-ratio estimate before computing the required "
    "trial count, verified by a dedicated unit test on the inlier ratio.",
    "Balancing SGBM parameters (block size, uniqueness ratio, speckle filtering) to maximize "
    "valid disparity coverage without introducing noisy mismatches in low-texture regions.",
    "Ensuring NaN/invalid pixels from unmatched disparity regions were handled consistently "
    "through to the final depth map and visualization colormap without runtime warnings.",
]))

# 13. Learnings & Key Takeaways
story.append(Paragraph("13. Learnings & Key Takeaways", styles["H1"]))
story.append(bullets([
    "Implementing RANSAC from scratch clarified why the rank-2 constraint on the Fundamental "
    "matrix and Sampson distance (rather than raw algebraic error) matter for real robustness.",
    "Small numerical-stability bugs (floating point underflow in adaptive trial-count formulas) "
    "can silently produce a badly under-fit model that still 'runs successfully' - reinforcing "
    "the value of explicit unit tests over eyeballing output images alone.",
    "Stereo rectification quality has an outsized effect on downstream disparity quality - "
    "epipolar geometry errors compound through the whole pipeline.",
]))

# 14. Future Enhancements
story.append(Paragraph("14. Future Enhancements", styles["H1"]))
story.append(bullets([
    "Replace simulated intrinsics with a real checkerboard-based camera calibration routine.",
    "Add an interactive Streamlit front-end for live parameter tuning.",
    "Extend to multi-view Structure-from-Motion with sparse 3D point-cloud visualization.",
    "GPU-accelerated SGBM (or a learned stereo-matching model) for real-time video depth.",
]))

# 15. References
story.append(Paragraph("15. References", styles["H1"]))
story.append(bullets([
    "R. Hartley and A. Zisserman, <i>Multiple View Geometry in Computer Vision</i>, Cambridge University Press.",
    "M. A. Fischler and R. C. Bolles, 'Random Sample Consensus: A Paradigm for Model Fitting with "
    "Applications to Image Analysis and Automated Cartography,' <i>Communications of the ACM</i>, 1981.",
    "D. Scharstein and R. Szeliski, Middlebury Stereo Vision Datasets.",
    "OpenCV documentation - Camera Calibration and 3D Reconstruction module.",
]))

doc = SimpleDocTemplate(
    REPORT_PATH, pagesize=A4,
    topMargin=1.5 * cm, bottomMargin=1.5 * cm, leftMargin=2 * cm, rightMargin=2 * cm
)
doc.build(story)
print("Report written to", REPORT_PATH)
