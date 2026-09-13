"""
utils.py
--------
Shared helper functions: image I/O, logging, and visualization utilities
used across the pipeline modules.
"""

import os
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger (module-level logging/monitoring)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_image(path: str, color: bool = True) -> np.ndarray:
    """Load an image from disk with validation and clear error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        raise ValueError(f"OpenCV failed to decode image: {path}")
    return img


def ensure_dir(path: str) -> None:
    """Create an output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_image(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write image to {path}")


def normalize_for_display(map_2d: np.ndarray) -> np.ndarray:
    """Normalize a float disparity/depth map to a viewable 8-bit image."""
    valid = map_2d[np.isfinite(map_2d) & (map_2d > 0)]
    if valid.size == 0:
        return np.zeros_like(map_2d, dtype=np.uint8)
    norm = np.nan_to_num(map_2d, nan=0.0)
    norm = np.clip(norm, 0, np.percentile(valid, 99))
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def save_colormap(path: str, map_2d: np.ndarray, colormap=cv2.COLORMAP_JET) -> None:
    """Save a float map as a pseudo-colored PNG for visual inspection."""
    display = normalize_for_display(map_2d)
    colored = cv2.applyColorMap(display, colormap)
    save_image(path, colored)


def side_by_side(img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
    """Horizontally stack two images (resizing to equal height if needed)."""
    h = min(img_left.shape[0], img_right.shape[0])
    left_r = cv2.resize(img_left, (int(img_left.shape[1] * h / img_left.shape[0]), h))
    right_r = cv2.resize(img_right, (int(img_right.shape[1] * h / img_right.shape[0]), h))
    return np.hstack([left_r, right_r])


def plot_histogram(data: np.ndarray, title: str, xlabel: str, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(data.flatten(), bins=60, color="#2A6F97")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
