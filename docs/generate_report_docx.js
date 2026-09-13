const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  PageBreak, BorderStyle, LevelFormat, convertInchesToTwip
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const DIAG = path.join(ROOT, "docs", "diagrams");
const OUT = path.join(ROOT, "outputs");
const metrics = JSON.parse(fs.readFileSync(path.join(OUT, "summary_metrics.json")));

function imgDims(filePath, maxWidthPx = 550) {
  // crude PNG dimension reader (avoids extra deps)
  const buf = fs.readFileSync(filePath);
  const w = buf.readUInt32BE(16);
  const h = buf.readUInt32BE(20);
  const scale = Math.min(1, maxWidthPx / w);
  return { width: Math.round(w * scale), height: Math.round(h * scale) };
}

function imageParagraph(filePath, maxWidthPx, captionText) {
  const { width, height } = imgDims(filePath, maxWidthPx);
  const children = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({ data: fs.readFileSync(filePath), transformation: { width, height }, type: "png" })],
      spacing: { before: 200, after: 80 },
    }),
  ];
  if (captionText) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: captionText, italics: true, size: 18, color: "5F5E5A" })],
      spacing: { after: 200 },
    }));
  }
  return children;
}

function heading1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function heading2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function body(text) {
  return new Paragraph({ children: [new TextRun({ text, size: 22 })], spacing: { after: 160 }, alignment: AlignmentType.JUSTIFIED });
}
function bullet(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 22 })],
    numbering: { reference: "bullet-list", level: 0 },
    spacing: { after: 80 },
  });
}

function metricsTable(rows) {
  const cellWidths = [5400, 3600];
  const headerRow = new TableRow({
    children: ["Metric", "Value"].map((t, i) => new TableCell({
      width: { size: cellWidths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "E6F1FB" },
      children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, size: 21 })] })],
    })),
  });
  const dataRows = rows.map(([k, v], idx) => new TableRow({
    children: [k, v].map((t, i) => new TableCell({
      width: { size: cellWidths[i], type: WidthType.DXA },
      shading: idx % 2 === 1 ? { type: ShadingType.CLEAR, fill: "F1EFE8" } : undefined,
      children: [new Paragraph({ children: [new TextRun({ text: String(t), size: 20 })] })],
    })),
  }));
  return new Table({
    width: { size: cellWidths[0] + cellWidths[1], type: WidthType.DXA },
    columnWidths: cellWidths,
    rows: [headerRow, ...dataRows],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullet-list",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.2) } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 } } }, // A4 in twips
    children: [
      // Cover page
      new Paragraph({ text: "", spacing: { before: 2400 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Stereo Vision Depth Estimator", bold: true, size: 48 })], spacing: { after: 200 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "A RANSAC-based Epipolar Geometry Pipeline for Dense Depth Estimation", size: 26, color: "444441" })], spacing: { after: 800 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Course: Computer Vision (CSE3010)", size: 24, color: "444441" })], spacing: { after: 80 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Submitted as part of: VITyarthi - Build Your Own Project", size: 24, color: "444441" })], spacing: { after: 400 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Harshit Agarwal (KAIZER)", size: 24, color: "444441" })], spacing: { after: 60 } }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "B.Tech CSE (AI/ML), VIT Bhopal University", size: 24, color: "444441" })] }),
      new Paragraph({ children: [new PageBreak()] }),

      // 1. Introduction
      heading1("1. Introduction"),
      body("Depth perception from images is a fundamental problem in computer vision with applications in robotics, autonomous navigation, and 3D reconstruction. This project implements a complete two-view (binocular) stereo vision pipeline that recovers a dense, per-pixel metric depth map from a stereo image pair, grounded directly in the projection models, epipolar geometry, and RANSAC concepts covered in the Computer Vision (CSE3010) syllabus."),

      // 2. Problem Statement
      heading1("2. Problem Statement"),
      body("A single 2D image discards depth information inherent to a 3D scene. Stereo vision recovers this depth by observing the same scene from two horizontally offset viewpoints: a 3D point projects to horizontally shifted pixel locations in the two images, and this shift (disparity) is inversely proportional to the point's distance from the camera. Recovering this relationship reliably requires (a) finding accurate correspondences between the two views, (b) rejecting incorrect matches robustly, and (c) using the recovered geometry to rectify and triangulate depth. This project addresses all three sub-problems end-to-end."),

      // 3. Functional Requirements
      heading1("3. Functional Requirements"),
      bullet("Module 1 - Calibration & Projection Model: maintain intrinsic camera matrix K, extrinsic transform [R|t], and 3x4 projection matrices for both views; undistort input frames."),
      bullet("Module 2 - Feature Matching & Epipolar Geometry: detect and match SIFT keypoints; robustly estimate the Fundamental matrix via a custom RANSAC implementation; derive the Essential matrix and recover relative camera pose."),
      bullet("Module 3 - Disparity & Depth Estimation: stereo-rectify the pair; compute a dense disparity map via Semi-Global Block Matching; convert disparity to metric depth."),
      bullet("Clear input/output structure: two image file paths in, a metric depth map plus diagnostic visualizations and a JSON metrics file out."),
      bullet("A logical, linear workflow: load -> match -> estimate geometry -> rectify -> disparity -> depth -> evaluate."),

      // 4. Non-functional Requirements
      heading1("4. Non-Functional Requirements"),
      bullet("Performance: full pipeline completes in under 4 seconds on a 1282x1110 stereo pair on CPU."),
      bullet("Reliability: adaptive-but-floored RANSAC trial scheduling prevents a single unlucky minimal sample from silently degrading the geometry estimate."),
      bullet("Error handling: explicit, typed exceptions (FileNotFoundError, ValueError, RuntimeError) with clear messages for missing files, undecodable images, and insufficient matches."),
      bullet("Logging/monitoring: every stage logs progress, match counts, inlier ratios, and disparity coverage via Python's logging module."),
      bullet("Maintainability: one responsibility per module, each independently unit-tested."),
      bullet("Resource efficiency: disparity computed once per run at native resolution; no redundant recomputation."),

      // 5. System Architecture
      heading1("5. System Architecture"),
      body("The system is organized as three sequential modules feeding an evaluation stage, all orchestrated by a single pipeline entry point (main.py). Figure 1 shows the high-level architecture."),
      ...imageParagraph(path.join(DIAG, "01_system_architecture.png"), 560, "Figure 1: System Architecture"),
      new Paragraph({ children: [new PageBreak()] }),

      // 6. Design Diagrams
      heading1("6. Design Diagrams"),
      heading2("6.1 Process Flow / Workflow Diagram"),
      ...imageParagraph(path.join(DIAG, "02_workflow.png"), 380, "Figure 2: Pipeline Workflow"),
      heading2("6.2 Use Case Diagram"),
      ...imageParagraph(path.join(DIAG, "03_use_case.png"), 500, "Figure 3: Use Case Diagram"),
      new Paragraph({ children: [new PageBreak()] }),
      heading2("6.3 Class / Component Diagram"),
      ...imageParagraph(path.join(DIAG, "04_class_diagram.png"), 560, "Figure 4: Class/Component Diagram"),
      heading2("6.4 Sequence Diagram"),
      ...imageParagraph(path.join(DIAG, "05_sequence_diagram.png"), 560, "Figure 5: Sequence Diagram of main.run_pipeline()"),
      new Paragraph({ children: [new PageBreak()] }),

      // 7. Design Decisions
      heading1("7. Design Decisions & Rationale"),
      bullet("Custom RANSAC over cv2.findFundamentalMat: implementing the normalized 8-point algorithm and Sampson-distance inlier scoring by hand demonstrates the algorithmic understanding required by the syllabus, rather than treating RANSAC as a black box."),
      bullet("SIFT over ORB: SIFT gives more stable, higher-quality correspondences on the textured Middlebury test scene, which matters more for geometry accuracy than raw speed here."),
      bullet("SGBM over simple block matching: Semi-Global Block Matching produces materially smoother, more complete disparity maps at a modest computational cost."),
      bullet("Modular package layout: calibration, epipolar geometry, and depth estimation are separated into independent modules so each can be unit-tested and reasoned about in isolation."),
      bullet("Simulated intrinsics: since no physical stereo rig/checkerboard was available for this coursework project, plausible fixed intrinsics were used; the code path supports swapping in real calibration output (via cv2.calibrateCamera) without any interface changes."),

      // 8. Implementation Details
      heading1("8. Implementation Details"),
      body("The pipeline is implemented in Python 3.10 using OpenCV for image processing and geometric primitives, and NumPy for the custom RANSAC / 8-point algorithm math. The Fundamental matrix is estimated by a from-scratch loop: (1) sample 8 random correspondences, (2) solve the normalized 8-point linear system via SVD, (3) enforce the rank-2 constraint by zeroing the smallest singular value, (4) score all correspondences by Sampson distance, (5) track the best inlier set, and (6) adaptively shrink the number of remaining trials using the standard RANSAC formula N = log(1-p) / log(1-w^s). The Essential matrix is derived as E = K^T F K, and cv2.recoverPose extracts the relative rotation and translation. These are fed to cv2.stereoRectify to produce a rectified pair on which cv2.StereoSGBM_create computes disparity, which is converted to metric depth via Z = (f * B) / d."),
      heading2("8.1 Repository Structure"),
      body("modules/config.py, modules/calibration.py, modules/epipolar.py, modules/depth.py, modules/evaluate.py, modules/visualize.py, modules/utils.py, main.py, tests/test_pipeline.py (8 source modules + 1 test module = 9 meaningful files, exceeding the minimum requirement)."),

      // 9. Dataset Description
      heading1("9. Dataset Description"),
      body("The pipeline was validated on the Middlebury 'Aloe' stereo pair (1282x1110, distributed with OpenCV's official sample data), a standard benchmark image pair for stereo matching research featuring a textured potted plant against a patterned background at multiple depths - ideal for visually validating disparity/depth separation. The pipeline accepts any horizontally-offset stereo pair as input via the --left/--right CLI arguments."),
      heading2("9.1 Model / Algorithm Selection Rationale"),
      body("RANSAC was chosen over least-squares fitting for the Fundamental matrix because feature matching inevitably produces outlier correspondences (repetitive texture, occlusion boundaries); a single incorrect match can arbitrarily corrupt a least-squares fit, while RANSAC's inlier-consensus scoring is robust to a majority-inlier match set. SGBM was chosen over simple block matching for its explicit smoothness penalty terms (P1, P2), which produce visibly cleaner disparity maps on textured scenes like the Aloe pair."),
      new Paragraph({ children: [new PageBreak()] }),

      // 10. Screenshots / Results
      heading1("10. Screenshots / Results"),
      ...imageParagraph(path.join(OUT, "02_feature_matches.png"), 560, "Figure 6: SIFT feature matches before RANSAC filtering"),
      ...imageParagraph(path.join(OUT, "03_ransac_inlier_matches.png"), 560, "Figure 7: RANSAC inlier matches (outliers rejected)"),
      new Paragraph({ children: [new PageBreak()] }),
      ...imageParagraph(path.join(OUT, "04_epipolar_lines.png"), 560, "Figure 8: Epipolar line overlay on the right image"),
      ...imageParagraph(path.join(OUT, "05_rectified_pair.png"), 560, "Figure 9: Stereo-rectified pair"),
      new Paragraph({ children: [new PageBreak()] }),
      ...imageParagraph(path.join(OUT, "06_disparity_map.png"), 420, "Figure 10: Disparity map (SGBM)"),
      ...imageParagraph(path.join(OUT, "07_depth_map.png"), 420, "Figure 11: Metric depth map"),

      heading2("10.1 Quantitative Results"),
      metricsTable([
        ["Total feature matches (post ratio-test)", metrics.num_matches],
        ["RANSAC inliers", metrics.num_ransac_inliers],
        ["RANSAC inlier ratio", `${metrics.quality_checks.ransac_inlier_ratio_pct}%`],
        ["Fundamental matrix rank", metrics.quality_checks.fundamental_matrix_rank],
        ["Mean epipolar constraint residual", metrics.epipolar_constraint_check.mean_abs_residual.toFixed(4)],
        ["Disparity valid coverage", `${metrics.quality_checks.disparity_valid_coverage_pct}%`],
        ["Depth valid pixels", `${metrics.quality_checks.depth_valid_pixels_pct}%`],
        ["Mean estimated depth", `${metrics.depth_statistics.mean_depth_m} m`],
        ["Pipeline runtime", `${metrics.runtime_sec} s`],
      ]),
      new Paragraph({ text: "", spacing: { after: 200 } }),

      // 11. Testing Approach
      heading1("11. Testing Approach"),
      body("9 automated pytest tests validate correctness at every pipeline stage: calibration matrix shapes, feature-matching output validity, the Fundamental matrix's mathematically-required rank-2 property, a sanity floor on RANSAC inlier ratio for a well-textured scene, the epipolar constraint residual (x2^T F x1 approx 0), disparity map coverage, the inverse disparity-depth relationship, correct NaN handling at zero disparity, and explicit error handling for missing input files. All 9 tests pass (see tests/test_pipeline.py)."),

      // 12. Challenges Faced
      heading1("12. Challenges Faced"),
      bullet("The initial adaptive RANSAC trial-count formula could collapse to a single iteration when an early minimal sample produced a very low (but non-zero) inlier count, due to floating-point underflow in w^8. Fixed by flooring the inlier-ratio estimate before computing the required trial count, verified by a dedicated unit test on the inlier ratio."),
      bullet("Balancing SGBM parameters (block size, uniqueness ratio, speckle filtering) to maximize valid disparity coverage without introducing noisy mismatches in low-texture regions."),
      bullet("Ensuring NaN/invalid pixels from unmatched disparity regions were handled consistently through to the final depth map and visualization colormap without runtime warnings."),

      // 13. Learnings
      heading1("13. Learnings & Key Takeaways"),
      bullet("Implementing RANSAC from scratch clarified why the rank-2 constraint on the Fundamental matrix and Sampson distance (rather than raw algebraic error) matter for real robustness."),
      bullet("Small numerical-stability bugs (floating point underflow in adaptive trial-count formulas) can silently produce a badly under-fit model that still 'runs successfully' - reinforcing the value of explicit unit tests over eyeballing output images alone."),
      bullet("Stereo rectification quality has an outsized effect on downstream disparity quality - epipolar geometry errors compound through the whole pipeline."),

      // 14. Future Enhancements
      heading1("14. Future Enhancements"),
      bullet("Replace simulated intrinsics with a real checkerboard-based camera calibration routine."),
      bullet("Add an interactive Streamlit front-end for live parameter tuning."),
      bullet("Extend to multi-view Structure-from-Motion with sparse 3D point-cloud visualization."),
      bullet("GPU-accelerated SGBM (or a learned stereo-matching model) for real-time video depth."),

      // 15. References
      heading1("15. References"),
      bullet("R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, Cambridge University Press."),
      bullet("M. A. Fischler and R. C. Bolles, \"Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography,\" Communications of the ACM, 1981."),
      bullet("D. Scharstein and R. Szeliski, Middlebury Stereo Vision Datasets."),
      bullet("OpenCV documentation - Camera Calibration and 3D Reconstruction module."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath = path.join(ROOT, "docs", "Project_Report.docx");
  fs.writeFileSync(outPath, buf);
  console.log("Wrote", outPath);
});
