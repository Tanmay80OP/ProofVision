"""Detect text-line regions with a horizontal projection profile on the OpenCV ink mask."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import ink_mask, to_gray


@dataclass(frozen=True)
class Region:
    id: str
    x: int
    y: int
    w: int
    h: int
    page_id: str = "page0"

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def detect_lines(img: np.ndarray, page_id: str = "page0", min_height: int = 12) -> list[Region]:
    mask = ink_mask(to_gray(img)) > 0
    H, W = mask.shape
    active = mask.sum(axis=1) > max(2, int(0.002 * W))
    bands: list[list[int]] = []
    start = None
    for y, a in enumerate(active):
        if a and start is None:
            start = y
        elif not a and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, H])
    gap = max(8, int(0.012 * H))  # merge fraction bars / exponents into their line
    merged: list[list[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    regions: list[Region] = []
    for y0, y1 in merged:
        if y1 - y0 < min_height:
            continue
        cols = np.where(mask[y0:y1].sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        regions.append(Region(f"L{len(regions) + 1}", x0, int(y0), x1 - x0, int(y1 - y0), page_id))
    return regions
