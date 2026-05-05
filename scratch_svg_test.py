import cv2
import numpy as np

def contour_to_svg_path(cnt, tension=0.4):
    pts = cnt.reshape(-1, 2).astype(float)
    n = len(pts)
    if n < 3:
        return ""
    
    x0, y0 = pts[0]
    path = [f"M {x0:.3f} {y0:.3f}"]
    
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        
        cp1 = p1 + tension * (p2 - p0) / 3.0
        cp2 = p2 - tension * (p3 - p1) / 3.0
        
        path.append(f"C {cp1[0]:.3f} {cp1[1]:.3f}, {cp2[0]:.3f} {cp2[1]:.3f}, {p2[0]:.3f} {p2[1]:.3f}")
        
    path.append("Z")
    return " ".join(path)

print(contour_to_svg_path(np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]])))
