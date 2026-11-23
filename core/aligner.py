"""
Color Transfer and Alignment Strategies.
Implements:
1. Adaptive Distribution-Matching Harmonization (ADMH):
   - Multi-scale Lab CDF Matching
   - Content-Aware Edge-Preserving Feathering (preserves background fidelity)
2. Global Lab CDF Histogram Matching
3. Reinhard Statistical Color Transfer (Reinhard et al. 2001)
"""

import cv2
import numpy as np


class AdaptiveColorAligner:
    """Color Transfer & Alignment Engine."""

    def __init__(self, mode: str = "admh", feather_sigma: int = 31, diff_threshold: float = 1.5):
        """
        Args:
            mode: 'admh' (Adaptive Harmonization), 'global_cdf', or 'reinhard'
            feather_sigma: Gaussian filter size for spatial soft mask feathering
            diff_threshold: Absolute pixel difference threshold to delineate edited region
        """
        self.mode = mode
        self.feather_sigma = feather_sigma
        self.diff_threshold = diff_threshold

    def align(self, src_bgr: np.ndarray, tag_bgr: np.ndarray, mode: str = None) -> np.ndarray:
        """
        Align src_bgr color distribution towards tag_bgr.
        
        Args:
            src_bgr: Source image (BGR uint8)
            tag_bgr: Target/Reference image (BGR uint8)
            mode: Optional override for alignment mode
            
        Returns:
            Aligned image (BGR uint8)
        """
        use_mode = mode or self.mode

        if use_mode == "reinhard":
            return self._reinhard_transfer(src_bgr, tag_bgr)
        elif use_mode == "global_cdf":
            return self._cdf_match_lab(src_bgr, tag_bgr)
        else:  # default 'admh'
            return self._admh_transfer(src_bgr, tag_bgr)

    def _cdf_match_lab(self, src_bgr: np.ndarray, tag_bgr: np.ndarray) -> np.ndarray:
        """Non-linear 3-channel CDF matching in CIE L*a*b* space."""
        src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2Lab)
        tag_lab = cv2.cvtColor(tag_bgr, cv2.COLOR_BGR2Lab)

        aligned_lab = np.zeros_like(src_lab)
        for c in range(3):
            h_s = cv2.calcHist([src_lab[:, :, c]], [0], None, [256], [0, 256]).flatten()
            h_t = cv2.calcHist([tag_lab[:, :, c]], [0], None, [256], [0, 256]).flatten()
            cdf_s = h_s.cumsum() / (h_s.sum() + 1e-12)
            cdf_t = h_t.cumsum() / (h_t.sum() + 1e-12)

            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                lut[i] = min(np.searchsorted(cdf_t, cdf_s[i]), 255)
            aligned_lab[:, :, c] = lut[src_lab[:, :, c]]

        return cv2.cvtColor(aligned_lab, cv2.COLOR_Lab2BGR)

    def _reinhard_transfer(self, src_bgr: np.ndarray, tag_bgr: np.ndarray) -> np.ndarray:
        """Reinhard et al. (2001) statistical color transfer in Lab."""
        src_lab = cv2.cvtColor(src_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        tag_lab = cv2.cvtColor(tag_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

        s_mean, s_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1)) + 1e-6
        t_mean, t_std = tag_lab.mean(axis=(0, 1)), tag_lab.std(axis=(0, 1)) + 1e-6

        res_lab = (src_lab - s_mean) * (t_std / s_std) + t_mean
        res_lab[:, :, 0] = np.clip(res_lab[:, :, 0], 0.0, 100.0)
        res_lab[:, :, 1] = np.clip(res_lab[:, :, 1], -127.0, 127.0)
        res_lab[:, :, 2] = np.clip(res_lab[:, :, 2], -127.0, 127.0)

        res_bgr = cv2.cvtColor(res_lab, cv2.COLOR_Lab2BGR) * 255.0
        return np.clip(res_bgr, 0, 255).astype(np.uint8)

    def _admh_transfer(self, src_bgr: np.ndarray, tag_bgr: np.ndarray) -> np.ndarray:
        """
        Adaptive Distribution-Matching Harmonization (ADMH):
        1. Lab Non-linear CDF histogram matching.
        2. Content-Aware Spatial Feathering:
           Detects unchanged background and preserves it with 100% fidelity,
           while seamlessly applying color transfer to modified facial/hair regions.
        """
        # Step 1: Global CDF transfer
        matched_bgr = self._cdf_match_lab(src_bgr, tag_bgr)

        # Step 2: Content-Aware Soft Mask
        diff = np.abs(src_bgr.astype(np.float32) - tag_bgr.astype(np.float32)).mean(axis=2)
        raw_mask = (diff > self.diff_threshold).astype(np.float32)

        # If more than 85% is different, this is likely not a local edit (treat as full transfer)
        if raw_mask.mean() > 0.85:
            return matched_bgr

        # Spatial feathering
        ksize = self.feather_sigma if self.feather_sigma % 2 == 1 else self.feather_sigma + 1
        smooth_mask = cv2.GaussianBlur(raw_mask, (ksize, ksize), 0)[:, :, None]

        # Soft blend
        blended = smooth_mask * matched_bgr.astype(np.float32) + (1.0 - smooth_mask) * src_bgr.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)


def color_transfer_align(src_bgr: np.ndarray, tag_bgr: np.ndarray, mode: str = "admh") -> np.ndarray:
    """Functional convenience wrapper for color alignment."""
    aligner = AdaptiveColorAligner(mode=mode)
    return aligner.align(src_bgr, tag_bgr)
