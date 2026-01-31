import cv2
import numpy as np

class ORBRansacMotionEstimator:
    def __init__(self, roi_mask: np.ndarray):
        self.roi_mask = roi_mask

        # ORB: fast binary descriptor
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=15
        )

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

    def reset(self):
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

    def process(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, self.roi_mask)

        # First frame init
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp, self.prev_des = kp, des
            return None

        # If not enough descriptors
        if self.prev_des is None or des is None or len(self.prev_des) < 50 or len(des) < 50:
            self.prev_gray = gray
            self.prev_kp, self.prev_des = kp, des
            return {"valid": False, "num_matches": 0, "inlier_ratio": 0.0}

        # KNN match + ratio test
        knn = self.bf.knnMatch(self.prev_des, des, k=2)
        good = []
        for m, n in knn:
            if m.distance < 0.85 * n.distance:
                good.append(m)

        if len(good) < 50:
            self.prev_gray = gray
            self.prev_kp, self.prev_des = kp, des
            return {"valid": False, "num_matches": len(good), "inlier_ratio": 0.0}

        # Build point arrays
        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 2)

        # Estimate affine (partial: rotation+translation+scale, no shear) with RANSAC
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99,
            maxIters=2000
        )

        if M is None or inliers is None:
            self.prev_gray = gray
            self.prev_kp, self.prev_des = kp, des
            return {"valid": False, "num_matches": len(good), "inlier_ratio": 0.0}

        inlier_count = int(inliers.sum())
        inlier_ratio = float(inlier_count / max(1, len(good)))

        # Translation in pixels (image motion from prev->curr)
        dx_px = float(M[0, 2])
        dy_px = float(M[1, 2])

        # Update prev
        self.prev_gray = gray
        self.prev_kp, self.prev_des = kp, des

        # Quality gate
        valid = (inlier_count >= 40) and (inlier_ratio >= 0.25)

        return {
            "valid": valid,
            "dx_px": dx_px,
            "dy_px": dy_px,
            "num_matches": len(good),
            "inlier_ratio": inlier_ratio,
            "inliers": inlier_count
        }
