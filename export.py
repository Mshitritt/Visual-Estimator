# CSV WRITER
import cv2
import csv
from pathlib import Path

from roi import get_video_meta, make_roi_mask
from motion_lk import LKMotionEstimator
from motion_orb import ORBRansacMotionEstimator
from geometry import intrinsics_from_hfov, px_to_speed_mps

VIDEO_PATH = r"videos/car overhead.mp4"

HFOV_DEG = 80.0
HEIGHT_M = 30.0
SMOOTH_WINDOW = 7  # later

def moving_average(xs, w):
    if w <= 1:
        return xs
    out = []
    s = 0.0
    q = []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def main():
    fps, w, h, n = get_video_meta(VIDEO_PATH)
    mask = make_roi_mask(w, h)

    fx, fy, cx, cy = intrinsics_from_hfov(w, h, HFOV_DEG)

    lk = LKMotionEstimator(mask)
    orb = ORBRansacMotionEstimator(mask)

    cap = cv2.VideoCapture(VIDEO_PATH)

    rows = []
    speed_lk_raw = []
    speed_orb_raw = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        lk_res = lk.process(frame)
        orb_res = orb.process(frame)

        lk_speed = None
        orb_speed = None
        lk_valid = False
        orb_valid = False
        lk_tracks = 0
        orb_inlier_ratio = 0.0

        if lk_res is not None and lk_res.get("dx_px") is not None:
            lk_valid = True
            lk_tracks = int(lk_res.get("num_tracks", 0))
            lk_speed = px_to_speed_mps(lk_res["dx_px"], lk_res["dy_px"], fx, fy, HEIGHT_M, fps)

        if orb_res is not None and orb_res.get("valid") is True:
            orb_valid = True
            orb_inlier_ratio = float(orb_res.get("inlier_ratio", 0.0))
            orb_speed = px_to_speed_mps(orb_res["dx_px"], orb_res["dy_px"], fx, fy, HEIGHT_M, fps)

        speed_lk_raw.append(lk_speed if lk_speed is not None else 0.0)
        speed_orb_raw.append(orb_speed if orb_speed is not None else 0.0)

        rows.append({
            "frame_idx": frame_idx,
            "time_sec": t,
            "speed_mps_lk_raw": lk_speed if lk_speed is not None else "",
            "speed_mps_orb_raw": orb_speed if orb_speed is not None else "",
            "valid_lk": int(lk_valid),
            "valid_orb": int(orb_valid),
            "tracks_lk": lk_tracks,
            "orb_inlier_ratio": orb_inlier_ratio,
        })

        frame_idx += 1

    cap.release()

    # Smooth (simple moving average over raw-with-zeros; later we can smooth only valid samples)
    lk_s = moving_average(speed_lk_raw, SMOOTH_WINDOW)
    orb_s = moving_average(speed_orb_raw, SMOOTH_WINDOW)

    for i in range(len(rows)):
        rows[i]["speed_mps_lk_smooth"] = lk_s[i]
        rows[i]["speed_mps_orb_smooth"] = orb_s[i]

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_csv = out_dir / "speed[export].csv"

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        wcsv.writerows(rows)

    print(f"Wrote: {out_csv}")
    print(f"Assumptions: HFOV={HFOV_DEG} deg, height={HEIGHT_M} m, fps={fps:.3f}, fx={fx:.1f}px")

if __name__ == "__main__":
    main()
