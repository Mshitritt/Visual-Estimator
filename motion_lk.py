import cv2
import numpy as np

class LKMotionEstimator:
    def __init__(self, roi_mask: np.ndarray):
        self.roi_mask = roi_mask

        # Shi–Tomasi params
        self.feature_params = dict(
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            mask=roi_mask
        )

        # Lucas–Kanade params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.prev_gray = None
        self.prev_pts = None

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None

    def process(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # First frame: just initialize
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            return None  # no motion yet

        if self.prev_pts is None or len(self.prev_pts) < 100:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            self.prev_gray = gray
            return None

        # Track points
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        good_prev = self.prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        if len(good_prev) < 20:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            return None

        # Pixel displacement
        flow = (good_next - good_prev).reshape(-1, 2)
        dx = flow[:, 0]
        dy = flow[:, 1]

        # Robust aggregation
        dx_med = float(np.median(dx))
        dy_med = float(np.median(dy))

        result = {
            "dx_px": dx_med,
            "dy_px": dy_med,
            "num_tracks": len(dx)
        }

        # Prepare for next frame
        self.prev_gray = gray
        self.prev_pts = good_next.reshape(-1, 1, 2)

        return result
