"""Recognition interface. Engines must label where their output came from."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class RecognitionResult:
    latex: str
    source: str                      # "manual" | "texteller" | ...
    quality: float | None = None     # None when the engine exposes no score
    quality_kind: str = "none"       # "model_probability" | "heuristic" | "none"


class Engine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> RecognitionResult: ...


class ManualEngine:
    """Fallback for tests/demos. Output is labeled source='manual' so it can never pass as OCR."""
    name = "manual"

    def __init__(self, lines: list[str]):
        self._lines, self._i = lines, 0

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        latex = self._lines[min(self._i, len(self._lines) - 1)]
        self._i += 1
        return RecognitionResult(latex=latex, source="manual")
