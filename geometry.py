# (pixels→meters)
import numpy as np
import math


def intrinsics_from_hfov(width_px: int, height_px: int, hfov_deg: float):
    hfov = math.radians(hfov_deg)
    fx = width_px / (2.0 * math.tan(hfov / 2.0))
    fy = fx  # reasonable default
    cx = width_px / 2.0
    cy = height_px / 2.0
    return fx, fy, cx, cy


def px_to_speed_mps(dx_px: float, dy_px: float, fx: float, fy: float, height_m: float, fps: float):
    # small-angle approximation on ground plane
    dx_m = height_m * (dx_px / fx)
    dy_m = height_m * (dy_px / fy)
    dist_m = math.sqrt(dx_m*dx_m + dy_m*dy_m)
    return dist_m * fps


"""
Assumptions:
-) focal length = 80
- principal point = 
    cx = width_image/2 pixels
    cy = height_image/ 2 pixels 

"""
