"""
Color Metrics & Color Space Evaluation Utilities.
Implements ISO/CIE standard CIEDE2000, Delta E 76, 3D Bhattacharyya distance,
discrete Wasserstein distance (Earth Mover's Distance), PSNR, SSIM, and gamut distributions.
"""

import cv2
import numpy as np


def ciede2000(lab1: np.ndarray, lab2: np.ndarray, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    """
    Vectorized ISO/CIE standard CIEDE2000 color difference formula.
    
    Args:
        lab1: NumPy array of shape (..., 3) with [L*, a*, b*]
        lab2: NumPy array of shape (..., 3) with [L*, a*, b*]
              L* in [0, 100], a* in [-128, 127], b* in [-128, 127]
        kL, kC, kH: Weighting factors (default 1.0)
        
    Returns:
        Array of Delta E 2000 values matching the leading spatial dimensions of the inputs.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0**7 + 1e-12)))

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = np.hypot(a1_prime, b1)
    C2_prime = np.hypot(a2_prime, b2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    dh_prime = h2_prime - h1_prime
    dh_prime = np.where(np.abs(h2_prime - h1_prime) <= 180.0, dh_prime,
               np.where(h2_prime <= h1_prime, dh_prime + 360.0, dh_prime - 360.0))
    dh_prime = np.where((C1_prime * C2_prime) == 0, 0.0, dh_prime)

    dH_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime * 0.5))

    L_bar_prime = 0.5 * (L1 + L2)
    C_bar_prime = 0.5 * (C1_prime + C2_prime)

    h_sum = h1_prime + h2_prime
    h_diff = np.abs(h1_prime - h2_prime)
    h_bar_prime = np.where(C1_prime * C2_prime == 0, h_sum,
                  np.where(h_diff <= 180.0, 0.5 * h_sum,
                  np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0))))

    T = (1.0 - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
             + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0)))

    d_theta = 30.0 * np.exp(-((h_bar_prime - 275.0) / 25.0)**2)
    C_bar_prime7 = C_bar_prime ** 7
    RC = 2.0 * np.sqrt(C_bar_prime7 / (C_bar_prime7 + 25.0**7 + 1e-12))
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC

    SL = 1.0 + (0.015 * (L_bar_prime - 50.0)**2) / np.sqrt(20.0 + (L_bar_prime - 50.0)**2)
    SC = 1.0 + 0.045 * C_bar_prime
    SH = 1.0 + 0.015 * C_bar_prime * T

    termL = dL_prime / (kL * SL)
    termC = dC_prime / (kC * SC)
    termH = dH_prime / (kH * SH)

    return np.sqrt(termL**2 + termC**2 + termH**2 + RT * termC * termH)


def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Euclidean distance in CIE L*a*b* space."""
    return np.sqrt(np.sum((lab1 - lab2)**2, axis=-1))


def bhattacharyya_3d(img1_lab_u8: np.ndarray, img2_lab_u8: np.ndarray, bins: int = 16) -> float:
    """
    3D Color histogram Bhattacharyya distance across L*, a*, b* channels.
    Higher values mean greater distributional divergence.
    """
    h1 = cv2.calcHist([img1_lab_u8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    h2 = cv2.calcHist([img2_lab_u8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    p1 = h1 / (h1.sum() + 1e-12)
    p2 = h2 / (h2.sum() + 1e-12)
    bc = np.sum(np.sqrt(p1 * p2))
    return float(-np.log(np.clip(bc, 1e-10, 1.0)))


def cosine_sim_3d(img1_lab_u8: np.ndarray, img2_lab_u8: np.ndarray, bins: int = 16) -> float:
    """3D Color histogram cosine similarity in [0, 1]."""
    h1 = cv2.calcHist([img1_lab_u8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]).flatten()
    h2 = cv2.calcHist([img2_lab_u8], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]).flatten()
    norm1 = np.linalg.norm(h1) + 1e-12
    norm2 = np.linalg.norm(h2) + 1e-12
    return float(np.dot(h1, h2) / (norm1 * norm2))


def fast_w1_channel(c1: np.ndarray, c2: np.ndarray, max_val: float = 100.0) -> float:
    """
    1D Wasserstein distance calculated via discrete cumulative distributions (CDF L1-norm).
    Provides exact Earth Mover's Distance in O(N) rather than O(N log N) sorting.
    """
    h1 = cv2.calcHist([c1], [0], None, [256], [0, 256]).flatten()
    h2 = cv2.calcHist([c2], [0], None, [256], [0, 256]).flatten()
    cdf1 = h1.cumsum() / (h1.sum() + 1e-12)
    cdf2 = h2.cumsum() / (h2.sum() + 1e-12)
    return float(np.sum(np.abs(cdf1 - cdf2)) * (max_val / 256.0))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Fast SSIM on luminance channel using Gaussian windowing."""
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    I1 = img1.astype(np.float64)
    I2 = img2.astype(np.float64)

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(I1, kernel, sigma)
    mu2 = cv2.GaussianBlur(I2, kernel, sigma)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(I1**2, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(I2**2, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(I1 * I2, kernel, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two uint8 BGR images."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(255.0**2 / mse))


def extract_histograms(img_bgr: np.ndarray, img_lab_u8: np.ndarray) -> dict:
    """
    Extract compact 32-bin 1D histograms and 16x16 2D a*-b* chromaticity grid.
    Suitable for fast real-time HTML5 canvas rendering on client side.
    """
    total = float(img_bgr.shape[0] * img_bgr.shape[1])

    # 32-bin RGB
    hist_r = cv2.calcHist([img_bgr], [2], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_bgr], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_bgr], [0], None, [32], [0, 256]).flatten()

    # 32-bin Lab
    hist_L = cv2.calcHist([img_lab_u8], [0], None, [32], [0, 256]).flatten()
    hist_a = cv2.calcHist([img_lab_u8], [1], None, [32], [0, 256]).flatten()
    hist_b_lab = cv2.calcHist([img_lab_u8], [2], None, [32], [0, 256]).flatten()

    # 2D chromaticity: a* vs b* (16x16 grid)
    hist_2d = cv2.calcHist([img_lab_u8], [1, 2], None, [16, 16], [0, 256, 0, 256])
    hist_2d_norm = (hist_2d / (total + 1e-12)).tolist()

    return {
        "rgb": {
            "r": (hist_r / total).round(4).tolist(),
            "g": (hist_g / total).round(4).tolist(),
            "b": (hist_b / total).round(4).tolist(),
        },
        "lab": {
            "L": (hist_L / total).round(4).tolist(),
            "a": (hist_a / total).round(4).tolist(),
            "b": (hist_b_lab / total).round(4).tolist(),
        },
        "gamut_2d": [[round(val, 5) for val in row] for row in hist_2d_norm],
    }


def generate_diff_heatmap(img1_bgr: np.ndarray, img2_bgr: np.ndarray, max_de: float = 15.0) -> np.ndarray:
    """
    Generates a perceptual difference heatmap using TURBO colormap overlay.
    """
    img1_lab = cv2.cvtColor(img1_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    img2_lab = cv2.cvtColor(img2_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    de_map = delta_e76(img1_lab, img2_lab)

    heat_u8 = np.clip(de_map / max_de * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    gray_bg = cv2.cvtColor(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray_bg, 0.35, heat_color, 0.65, 0)
    return overlay
