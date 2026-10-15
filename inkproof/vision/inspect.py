"""Targeted crops with coordinate mapping back to the original page. The agent calls these."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import make_variants, to_gray
from .regions import Region


@dataclass
class Crop:
    crop_id: str
    region_id: str
    page_id: str
    bbox: tuple[int, int, int, int]  # in ORIGINAL page coordinates (after padding/clamping)
    variant: str
    zoom: float
    attempt: int
    image: np.ndarray

    def meta(self) -> dict:
        return {k: getattr(self, k) for k in ("crop_id", "region_id", "page_id", "bbox", "variant", "zoom", "attempt")}


def make_crop(page: np.ndarray, region: Region, variant: str = "gray", zoom: float = 1.0,
              pad: int = 10, attempt: int = 1) -> Crop:
    gray = to_gray(page)
    H, W = gray.shape
    x0, y0 = max(0, region.x - pad), max(0, region.y - pad)
    x1, y1 = min(W, region.x + region.w + pad), min(H, region.y + region.h + pad)
    sub = gray[y0:y1, x0:x1]
    variants = make_variants(sub)
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {sorted(variants)}.")
    out = variants[variant]
    if zoom != 1.0:
        out = cv2.resize(out, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    return Crop(f"{region.id}-{variant}-z{zoom:g}-a{attempt}", region.id, region.page_id,
                (x0, y0, x1 - x0, y1 - y0), variant, zoom, attempt, out)
