"""
generate_diagrams.py
---------------------
Generates all design-documentation diagrams required by section 4 of the
VITyarthi project spec: System Architecture, Process Flow / Workflow,
Use Case, Class/Component, and Sequence diagrams.

Run:  python3 docs/generate_diagrams.py
Outputs PNGs into docs/diagrams/
"""

import os
import graphviz

OUT = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(OUT, exist_ok=True)

COMMON_ATTRS = dict(
    fontname="Helvetica", fontsize="11", rankdir="TB"
)


def render(g: graphviz.Digraph, name: str):
    path = g.render(filename=name, directory=OUT, format="png", cleanup=True)
    print("Wrote", path)


# ---------------------------------------------------------------------------
# 1. System Architecture Diagram
# ---------------------------------------------------------------------------
def system_architecture():
    g = graphviz.Digraph("architecture", graph_attr={"rankdir": "LR", "splines": "ortho", "fontsize": "12"})
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11")

    g.node("input", "Stereo Image\nPair (L/R)", fillcolor="#E6F1FB")
    with g.subgraph(name="cluster_pipeline") as c:
        c.attr(label="Stereo Vision Depth Estimation Pipeline", style="rounded", fontsize="12")
        c.node("calib", "Module 1\nCalibration &\nProjection Model", fillcolor="#EAF3DE")
        c.node("epi", "Module 2\nFeature Matching +\nRANSAC Epipolar Geometry", fillcolor="#FAEEDA")
        c.node("depthm", "Module 3\nDisparity & Depth\nEstimation (SGBM)", fillcolor="#FBEAF0")
        c.edge("calib", "epi")
        c.edge("epi", "depthm")

    g.node("eval", "Evaluation &\nQuality Metrics", shape="box", fillcolor="#F1EFE8")
    g.node("output", "Depth Map +\nVisual Outputs +\nsummary_metrics.json", fillcolor="#E1F5EE")

    g.edge("input", "calib")
    g.edge("depthm", "eval")
    g.edge("depthm", "output")
    g.edge("eval", "output")

    render(g, "01_system_architecture")


# ---------------------------------------------------------------------------
# 2. Process Flow / Workflow Diagram
# ---------------------------------------------------------------------------
def workflow():
    g = graphviz.Digraph("workflow", graph_attr={"rankdir": "TB", "fontsize": "12"})
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#E6F1FB", fontname="Helvetica", fontsize="11")

    steps = [
        ("s1", "Load left/right images\n+ undistort"),
        ("s2", "Detect SIFT keypoints\n& match (Lowe ratio test)"),
        ("s3", "Estimate Fundamental matrix\nvia custom RANSAC loop"),
        ("s4", "Derive Essential matrix\n& recover relative pose (R, t)"),
        ("s5", "Stereo-rectify image pair"),
        ("s6", "Compute dense disparity map\n(Semi-Global Block Matching)"),
        ("s7", "Convert disparity -> depth\n(Z = f * B / d)"),
        ("s8", "Save visual outputs +\nevaluation metrics (JSON)"),
    ]
    for node_id, label in steps:
        g.node(node_id, label)
    for a, b in zip([s[0] for s in steps], [s[0] for s in steps][1:]):
        g.edge(a, b)

    render(g, "02_workflow")


# ---------------------------------------------------------------------------
# 3. Use Case Diagram (simplified, actor + system boundary)
# ---------------------------------------------------------------------------
def use_case():
    g = graphviz.Digraph("usecase", graph_attr={"rankdir": "LR", "fontsize": "12"})
    g.node("student", "Student /\nEvaluator", shape="ellipse", style="filled", fillcolor="#EEEDFE")

    with g.subgraph(name="cluster_system") as c:
        c.attr(label="Stereo Depth Estimator System", style="rounded", fontsize="12")
        c.attr("node", shape="ellipse", style="filled", fillcolor="#E1F5EE", fontname="Helvetica", fontsize="10")
        c.node("uc1", "Upload stereo\nimage pair")
        c.node("uc2", "Run calibration &\nrectification")
        c.node("uc3", "View RANSAC\ninlier matches")
        c.node("uc4", "View epipolar\nline overlay")
        c.node("uc5", "View disparity /\ndepth map")
        c.node("uc6", "Inspect evaluation\nmetrics (JSON)")
        c.node("uc7", "Run automated\nunit tests")

    for uc in ["uc1", "uc2", "uc3", "uc4", "uc5", "uc6", "uc7"]:
        g.edge("student", uc)

    render(g, "03_use_case")


# ---------------------------------------------------------------------------
# 4. Class / Component Diagram
# ---------------------------------------------------------------------------
def class_diagram():
    g = graphviz.Digraph("class", graph_attr={"rankdir": "TB", "fontsize": "12"})
    g.attr("node", shape="record", fontname="Helvetica", fontsize="10", style="filled", fillcolor="#F1EFE8")

    g.node("config", "{config|+ K : ndarray\l+ DIST_COEFFS\l+ FOCAL_LENGTH_PX\l+ BASELINE_M\l+ RANSAC_* params\l}")
    g.node("calibration", "{StereoCalibration|+ K, dist, baseline_m\l+ P_left, P_right\l|+ projection_matrices()\l+ undistort(img)\l+ rectify_pair(L, R, R_rel, t_rel)\l}")
    g.node("epipolar", "{epipolar (module)|+ detect_and_match_features()\l+ estimate_fundamental_ransac()\l+ recover_relative_pose()\l+ compute_epipolar_lines()\l}")
    g.node("depth", "{depth (module)|+ compute_disparity()\l+ disparity_to_depth()\l+ depth_statistics()\l}")
    g.node("evaluate", "{evaluate (module)|+ epipolar_constraint_error()\l+ inlier_ratio()\l+ disparity_coverage()\l+ run_unit_style_checks()\l}")
    g.node("visualize", "{visualize (module)|+ draw_matches()\l+ draw_epilines()\l}")
    g.node("utils", "{utils (module)|+ load_image()\l+ save_image()\l+ save_colormap()\l+ normalize_for_display()\l}")
    g.node("main", "{main.py|+ run_pipeline(left, right, outdir)\l+ parse_args()\l}")

    g.edge("main", "calibration")
    g.edge("main", "epipolar")
    g.edge("main", "depth")
    g.edge("main", "evaluate")
    g.edge("main", "visualize")
    g.edge("main", "utils")
    g.edge("calibration", "config")
    g.edge("epipolar", "config")
    g.edge("depth", "config")

    render(g, "04_class_diagram")


# ---------------------------------------------------------------------------
# 5. Sequence Diagram (main pipeline call sequence)
# ---------------------------------------------------------------------------
def sequence_diagram():
    g = graphviz.Digraph("sequence", graph_attr={"rankdir": "LR", "fontsize": "11", "splines": "ortho"})
    g.attr("node", shape="box", style="filled", fillcolor="#E6F1FB", fontname="Helvetica", fontsize="10")

    order = ["User", "main.py", "calibration", "epipolar", "depth", "evaluate", "utils"]
    for n in order:
        g.node(n)
    edges = [
        ("User", "main.py", "run"),
        ("main.py", "utils", "load_image()"),
        ("main.py", "calibration", "undistort()"),
        ("main.py", "epipolar", "detect_and_match_features()"),
        ("main.py", "epipolar", "estimate_fundamental_ransac()"),
        ("main.py", "epipolar", "recover_relative_pose()"),
        ("main.py", "calibration", "rectify_pair()"),
        ("main.py", "depth", "compute_disparity()"),
        ("main.py", "depth", "disparity_to_depth()"),
        ("main.py", "evaluate", "run_unit_style_checks()"),
        ("main.py", "utils", "save_colormap() / save_image()"),
        ("main.py", "User", "summary_metrics.json"),
    ]
    for a, b, label in edges:
        g.edge(a, b, label=label, fontsize="9")

    render(g, "05_sequence_diagram")


if __name__ == "__main__":
    system_architecture()
    workflow()
    use_case()
    class_diagram()
    sequence_diagram()
