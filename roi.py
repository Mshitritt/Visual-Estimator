# mask creation
import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = r"videos/car overhead.mp4"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_roi_mask(w: int, h: int) -> np.ndarray:
    """
    Mask design for this clip:
    - Keep a vertical band centered on the road (reduces tree parallax).
    - Exclude a central box where the car is (avoid moving-object flow).
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    # Road band (tuneable):
    x0 = int(0.32 * w)
    x1 = int(0.68 * w)
    mask[:, x0:x1] = 255

    # Exclude car region (center-ish box):
    cx = int(0.50 * w)
    cy = int(0.52 * h)
    box_w = int(0.22 * w)
    box_h = int(0.18 * h)

    x0c = max(0, cx - box_w // 2)
    x1c = min(w, cx + box_w // 2)
    y0c = max(0, cy - box_h // 2)
    y1c = min(h, cy + box_h // 2)

    mask[y0c:y1c, x0c:x1c] = 0
    return mask


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame_bgr.copy()

    green = np.zeros_like(frame_bgr)
    green[:, :, 1] = 180  # green channel

    alpha = 0.35

    blended = (overlay * (1 - alpha) + green * alpha).astype(np.uint8)

    m2 = mask > 0  # shape (H, W)
    overlay[m2] = blended[m2]  # works: selects pixel rows of shape (N,3)

    return overlay


def get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n

def main():
    fps, w, h, n = get_video_meta(VIDEO_PATH)
    print(f"Video: {w}x{h}, fps={fps:.3f}, frames={n}, duration={n/fps:.2f}s")

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    mask = make_roi_mask(w, h)
    vis = overlay_mask(frame0, mask)

    cv2.imwrite(str(OUT_DIR / "frame0.png"), frame0)
    cv2.imwrite(str(OUT_DIR / "roi_mask.png"), mask)
    cv2.imwrite(str(OUT_DIR / "frame0_roi_overlay.png"), vis)

    print("Wrote outputs/")
    print(" - outputs/frame0.png")
    print(" - outputs/roi_mask.png")
    print(" - outputs/frame0_roi_overlay.png")


if __name__ == "__main__":
    main()
