MM_TO_PTS = 2.8346  # 1mm = 72/25.4 pts


def mm_to_pts(mm: float) -> float:
    return mm * MM_TO_PTS


def pts_to_mm(pts: float) -> float:
    return pts / MM_TO_PTS


def px_to_mm(px: int, dpi: float = 96.0) -> float:
    return px * 25.4 / dpi


def mm_to_px(mm: float, dpi: float = 96.0) -> int:
    return int(mm * dpi / 25.4)


def eps_bounding_box(width_mm: float, height_mm: float) -> str:
    """Return the %%BoundingBox line for an EPS header."""
    w = int(mm_to_pts(width_mm))
    h = int(mm_to_pts(height_mm))
    return f"%%BoundingBox: 0 0 {w} {h}"
