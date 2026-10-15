"""Slice 1 demo: synthetic page -> OpenCV 5 line detection -> annotated image + verification table."""
import cv2, numpy as np
from inkproof.vision.regions import detect_lines
from inkproof.vision.preprocess import ink_stats, to_gray
from inkproof.reasoning.verifier import verify_step

LINES = ["2(x+3)=14", "2x+6=14", "2x=8", "x=5"]
LATEX = ["2(x+3)=14", "2x+6=14", "2x=8", "x=5"]   # manual labels until the OCR engine is wired in
COLORS = {"VERIFIED": (60, 160, 60), "POTENTIAL_ERROR": (60, 60, 220), "UNCERTAIN": (30, 150, 230), "UNSUPPORTED": (150, 150, 150)}

img = np.full((120 * len(LINES) + 40, 700, 3), 255, np.uint8)
for i, t in enumerate(LINES):
    cv2.putText(img, t, (40, 100 + 120 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)

print("OpenCV", cv2.__version__)
regions = detect_lines(img)
out = img.copy()
for i, r in enumerate(regions):
    status = "START" if i == 0 else None
    if i:
        res = verify_step(LATEX[i - 1], LATEX[i]); status = res.status.value
        print(f"{r.id} {LATEX[i]:<10} {status:<16} {res.reason} {res.evidence}")
    col = COLORS.get(status, (120, 120, 120))
    cv2.rectangle(out, (r.x - 8, r.y - 8), (r.x + r.w + 8, r.y + r.h + 8), col, 2)
    cv2.putText(out, status, (r.x + r.w + 20, r.y + r.h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
print("stats L3:", {k: round(v, 3) for k, v in ink_stats(to_gray(img)[regions[2].y:regions[2].y + regions[2].h]).items()})
cv2.imwrite("samples/annotated_demo.png", out)
