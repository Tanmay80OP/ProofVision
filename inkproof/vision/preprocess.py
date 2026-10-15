"""OpenCV 5 preprocessing: validation, perspective correction, variants, ink statistics."""
from __future__ import annotations

import cv2
import numpy as np

MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 100, 8000


def decode_image(data: bytes) -> np.ndarray:
    """Validate and decode an uploaded image. Raises ValueError with a user-safe message."""
    if len(data) > MAX_BYTES:
        raise ValueError("Image is larger than 10 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image file.")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise ValueError(f"Image side must be between {MIN_SIDE} and {MAX_SIDE} px.")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def correct_perspective(img: np.ndarray, min_area_frac: float = 0.4) -> tuple[np.ndarray, bool]:
    """Warp a photographed page to a top-down view. Returns (image, was_corrected)."""
    blur = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 50, 150), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, False
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4 or cv2.contourArea(approx) < min_area_frac * img.shape[0] * img.shape[1]:
        return img, False
    tl, tr, br, bl = _order_corners(approx.reshape(4, 2).astype(np.float32))
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h)), True


def make_variants(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Alternative views of the same pixels. Binarization is NOT assumed to help recognition."""
    den = cv2.fastNlMeansDenoising(gray, None, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
    adaptive = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    _, otsu = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"gray": gray, "clahe": clahe, "adaptive": adaptive, "otsu": otsu}


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask where ink = 255 (used for layout analysis, not for recognition)."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    m = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def ink_stats(gray: np.ndarray) -> dict[str, float]:
    """Visual evidence the agent reasons over: sharpness, ink density, mean stroke width."""
    mask = ink_mask(gray)
    ink = mask > 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "ink_ratio": float(ink.mean()),
        "stroke_width": float(2 * dist[ink].mean()) if ink.any() else 0.0,
        "contrast": float(gray.std()),
    }
