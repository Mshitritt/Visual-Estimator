import csv
import os
import cv2


VIDEO_IN  = r"../videos/car overhead.mp4"
CSV_LK    = r"../outputs/speeds_LK.csv"
CSV_ORB   = r"../outputs/speeds_ORB.csv"
VIDEO_OUT = r"../outputs/video_with_speed.mp4"

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_speed_map(csv_path: str):
    """
    Returns dict: frame_idx -> (speed_mps or None, ok)
    """
    speed_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["frame_idx"])
            ok = int(row.get("ok", "1") or "1")
            sp_raw = row.get("speed_mps", "")
            speed = float(sp_raw) if sp_raw not in ("", None) else None
            speed_map[fi] = (speed, ok)
    return speed_map


def main():
    os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)

    lk_map  = load_speed_map(CSV_LK)
    orb_map = load_speed_map(CSV_ORB)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lk_speed,  lk_ok  = lk_map.get(frame_idx,  (None, 0))
        orb_speed, orb_ok = orb_map.get(frame_idx, (None, 0))

        # Compose text lines
        if lk_speed is None:
            lk_text = "LK  : N/A"
        else:
            lk_text = f"LK  : {lk_speed:6.2f} m/s"

        if orb_speed is None:
            orb_text = "ORB : N/A"
        else:
            orb_text = f"ORB : {orb_speed:6.2f} m/s"

        # Background box
        lines = [lk_text, orb_text]
        font_scale = 0.8
        thickness = 2

        max_w = 0
        total_h = 0
        for line in lines:
            (tw, th), bl = cv2.getTextSize(line, FONT, font_scale, thickness)
            max_w = max(max_w, tw)
            total_h += th + bl + 10

        x, y = 20, 40
        cv2.rectangle(
            frame,
            (x - 10, y - total_h),
            (x + max_w + 10, y + 10),
            (0, 0, 0),
            -1
        )

        # Draw text
        yy = y
        cv2.putText(frame, lk_text,  (x, yy), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        yy += 30
        cv2.putText(frame, orb_text, (x, yy), FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    out.release()
    cap.release()
    print(f"Saved: {VIDEO_OUT}")


if __name__ == "__main__":
    main()