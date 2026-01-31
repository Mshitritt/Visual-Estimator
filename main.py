import cv2
import numpy as np

from geometry import intrinsics_from_hfov, px_to_speed_mps
from roi import make_roi_mask
from motion_orb import ORBRansacMotionEstimator
from motion_lk import LKMotionEstimator
from csv_writer import CsvWriter

VIDEO_PATH = r"videos/car overhead.mp4"
HFOV_DEG = 80.0
HEIGHT_M = 30.0

# choose here (or later via CLI): "ORB" or "LK"
ESTIMATOR_NAME = "ORB"

BASE_HEADER = ["frame_idx", "t_sec", "ok", "dx_px", "dy_px", "speed_mps"]

ESTIMATORS = {
    "ORB": {
        "cls": ORBRansacMotionEstimator,
        "csv": "outputs/speeds_ORB.csv",
        "extra_header": ["inlier_ratio", "num_matches", "inliers"],  # add more if your ORB returns them
    },
    "LK": {
        "cls": LKMotionEstimator,
        "csv": "outputs/speeds_LK.csv",
        "extra_header": ["num_tracks"],  # LK returns this
    }
}
# for debug
prev_frame_bgr = None
prev_frame_idx = None


def main():
    cfg = ESTIMATORS[ESTIMATOR_NAME]
    out_csv = cfg["csv"]
    header = BASE_HEADER + cfg["extra_header"]

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 1) intrinsics
    fx, fy, cx, cy = intrinsics_from_hfov(w, h, HFOV_DEG)
    print("HFOV_DEG = ", HFOV_DEG)
    print("fx = ", fx)
    print("fy = ", fy)
    print("cx = ", cx)
    print("cy = ", cy)

    # 2) preprocessing setup
    roi_mask = make_roi_mask(w, h)

    # 3) estimator object (selected)
    estimator = cfg["cls"](roi_mask)
    estimator.reset()

    # 5) writer init (selected)
    writer = CsvWriter(out_csv, header=header)
    writer.open()

    # 4) loop
    frame_idx = 0
    prev_speed = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray_roi = cv2.bitwise_and(frame_gray, frame_gray, mask=roi_mask)
        frame_for_est = cv2.cvtColor(frame_gray_roi, cv2.COLOR_GRAY2BGR)

        res = estimator.process(frame_for_est)  # dict or None

        ok = False
        dx_px = dy_px = 0.0

        if res is not None:
            ok = bool(res.get("valid", True))  # ORB has 'valid', LK doesn't
            dx_px = float(res.get("dx_px", 0.0))
            dy_px = float(res.get("dy_px", 0.0))

        speed_mps = 0.0
        if ok:
            speed_mps = px_to_speed_mps(dx_px, dy_px, fx, fy, HEIGHT_M, fps)
        # ---------- DEBUG SNAPSHOT ----------
        # if ok and speed_mps == 0.0 and prev_frame_bgr is not None:
        #     cv2.imwrite(f"frames/drop_{frame_idx}_prev.png", prev_frame_bgr)
        #     cv2.imwrite(f"frames/drop_{frame_idx}_curr.png", frame_bgr)
        # -----------------------------------

        t_sec = frame_idx / fps
        # POLICY - for duplicate frames, keep the previous speed.
        if speed_mps < 1 and frame_idx > 1:
            speed_mps = prev_speed
        # build a stable row schema
        row = {
            "frame_idx": frame_idx,
            "t_sec": t_sec,
            "ok": int(ok),
            "dx_px": dx_px,
            "dy_px": dy_px,
            "speed_mps": speed_mps,
        }

        # merge estimator-specific fields (safe even if different)
        if isinstance(res, dict):
            row.update(res)

        writer.write_row(row)

        # update previous frame AFTER everything
        prev_frame_bgr = frame_bgr.copy()
        prev_frame_idx = frame_idx
        prev_speed = speed_mps
        frame_idx += 1

    writer.close()
    cap.release()


if __name__ == "__main__":
    main()
