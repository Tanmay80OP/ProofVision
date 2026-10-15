import cv2, numpy as np, pytest
from inkproof.vision.inspect import make_crop
from inkproof.vision.preprocess import correct_perspective, decode_image, ink_stats, make_variants, to_gray
from inkproof.vision.regions import detect_lines


def page(lines=("2(x+3)=14", "2x+6=14", "2x=8", "x=4")):
    img = np.full((120 * len(lines) + 40, 700, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (40, 100 + 120 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
    return img


def test_opencv_is_v5():
    assert cv2.__version__.startswith("5."), cv2.__version__


def test_detect_lines_count_and_order():
    regs = detect_lines(page())
    assert len(regs) == 4
    assert [r.y for r in regs] == sorted(r.y for r in regs)


def test_crop_metadata_and_variants():
    img = page()
    r = detect_lines(img)[1]
    c = make_crop(img, r, variant="clahe", zoom=2.0, attempt=2)
    assert c.attempt == 2 and c.variant == "clahe"
    assert c.image.shape[0] > r.h * 1.5
    x, y, w, h = c.bbox
    assert 0 <= x and 0 <= y and x + w <= img.shape[1] and y + h <= img.shape[0]
    with pytest.raises(ValueError):
        make_crop(img, r, variant="nope")


def test_ink_stats_keys_and_values():
    s = ink_stats(to_gray(page()))
    assert set(s) == {"sharpness", "ink_ratio", "stroke_width", "contrast"}
    assert 0 < s["ink_ratio"] < 0.5 and s["stroke_width"] > 0


def test_blur_lowers_sharpness():
    g = to_gray(page())
    assert ink_stats(cv2.GaussianBlur(g, (21, 21), 0))["sharpness"] < ink_stats(g)["sharpness"]


def test_variant_names():
    assert set(make_variants(to_gray(page()))) == {"gray", "clahe", "adaptive", "otsu"}


def test_perspective_correction_on_skewed_page():
    flat = page()
    h, w = flat.shape[:2]
    canvas = np.full((h + 200, w + 300, 3), 90, np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[120, 60], [w + 150, 20], [w + 200, h + 150], [60, h + 120]])
    warped = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(src, dst), (w + 300, h + 200),
                                 dst=canvas.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    out, ok = correct_perspective(warped)
    assert ok and abs(out.shape[1] / out.shape[0] - w / h) < 0.35


def test_decode_rejects_bad_input():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError):
        decode_image(b"0" * (11 * 1024 * 1024))
    ok, buf = cv2.imencode(".png", page())
    assert decode_image(buf.tobytes()).ndim == 3
